import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from openai import OpenAI

# Cargar variables de entorno
load_dotenv()

client = OpenAI()

# Configuración de la página
st.set_page_config(page_title="Sistema de Recomendación con LLM", layout="wide")


def main():
    # Título de la aplicación
    st.title("Sistema de Recomendación de películas")

    # Cargar dataset
    ratings = load_movielens_data()

    # Estadísticas
    st.subheader("Ficha técnica del dataset")
    html_table = """
    <table style="width:100%; border-collapse: collapse; text-align: left;">
        <tr><td><b>Nombre del dataset</b></td><td>MovieLens 100K</td></tr>
        <tr><td><b>Cantidad de observaciones</b></td><td>{}</td></tr>
        <tr><td><b>Usuarios únicos</b></td><td>{}</td></tr>
        <tr><td><b>Películas únicas</b></td><td>{}</td></tr>
    </table>
    """.format(
        len(ratings),
        ratings["userId"].nunique(),
        ratings["movieId"].nunique(),
    )
    st.markdown(html_table, unsafe_allow_html=True)

    # Mostrar las primeras 10 filas del dataset original
    st.subheader("Vista previa del dataset original")
    st.write("Estas son las primeras 10 filas del dataset antes del procesamiento:")
    st.dataframe(ratings.head(10), use_container_width=True)

    # Visualizaciones adicionales
    display_insights(ratings)

    # Generar matriz usuario-item
    user_item_matrix = generate_user_item_matrix(ratings)

    # Mostrar la matriz usuario-item (solo las primeras 10 filas)
    st.subheader("Matriz Usuario-Item")
    st.write(
        "La matriz usuario-item muestra cómo los usuarios califican las películas. A continuación se presentan las primeras 10 filas."
    )
    st.dataframe(user_item_matrix.head(10))

    # Configuración del agente inteligente en el sidebar
    st.sidebar.header("Agente Inteligente: Sugerencias Personalizadas")
    selected_user = st.sidebar.selectbox(
        "Selecciona el ID de un usuario para recomendaciones", user_item_matrix.index
    )
    if st.sidebar.button("Obtener sugerencias"):
        intelligent_agent(selected_user, ratings, user_item_matrix)

    # Configuración del método de reducción y botón
    st.subheader("Configuración de reducción de dimensionalidad")
    method = st.selectbox("Selecciona el método:", ["PCA", "t-SNE"])
    if st.button("Calcular y graficar embeddings"):
        embeddings = reduce_dimensionality(user_item_matrix, method)
        plot_embeddings(embeddings, method)

        # Mostrar tabla con top 10 usuarios similares
        st.subheader(f"Top 10 usuarios más similares al Usuario {selected_user}")
        top_similar_users = get_top_similar_users(user_item_matrix, selected_user)
        st.write("Usuarios más similares:")
        st.dataframe(top_similar_users, use_container_width=True)


@st.cache_data
def load_movielens_data():
    """Carga el dataset de MovieLens y los nombres de las películas."""
    # Cargar ratings
    url_ratings = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
    column_names = ["userId", "movieId", "rating", "timestamp"]
    ratings = pd.read_csv(url_ratings, sep="\t", names=column_names)

    # Cargar nombres de películas
    url_movies = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
    movie_columns = [
        "movieId",
        "title",
        "release_date",
        "video_release_date",
        "IMDb_URL",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]
    movies = pd.read_csv(
        url_movies,
        sep="|",
        encoding="latin-1",
        names=movie_columns,
        usecols=[0, 1],
    )
    movies["movieId"] = movies["movieId"].astype(int)

    # Unir ratings con nombres de películas
    ratings = pd.merge(ratings, movies, on="movieId")

    return ratings


@st.cache_data
def generate_user_item_matrix(data):
    """Genera la matriz usuario-item."""
    user_item_matrix = data.pivot(index="userId", columns="movieId", values="rating")
    user_item_matrix.fillna(0, inplace=True)
    return user_item_matrix


@st.cache_data
def reduce_dimensionality(matrix, method):
    """Reduce la dimensionalidad de la matriz usuario-item."""
    if method == "PCA":
        reducer = PCA(n_components=2)
    elif method == "t-SNE":
        reducer = TSNE(n_components=2, random_state=42)
    embeddings = reducer.fit_transform(matrix.values)
    return pd.DataFrame(embeddings, columns=["dim_1", "dim_2"], index=matrix.index)


def plot_embeddings(embeddings, method):
    """Genera una visualización de los embeddings en 2D."""
    st.header("Visualización de Embeddings")
    fig = px.scatter(
        embeddings,
        x="dim_1",
        y="dim_2",
        text=embeddings.index,
        title=f"Visualización de Usuarios ({method})",
    )
    st.plotly_chart(fig)


def intelligent_agent(selected_user, ratings, user_item_matrix):
    """Genera sugerencias personalizadas basadas en el dataset y en GPT-4, y escribe resultados en el sidebar."""
    st.sidebar.subheader(f"Sugerencias para el Usuario {selected_user}")

    # Obtener películas mejor calificadas por el usuario
    user_ratings = ratings[ratings["userId"] == selected_user]
    favorite_movies = user_ratings[user_ratings["rating"] >= 4]["movieId"].unique()
    favorite_movie_titles = user_ratings[user_ratings["rating"] >= 4]["title"].unique()

    # Verificar si el usuario tiene películas favoritas
    if len(favorite_movies) == 0:
        st.sidebar.write(
            "El usuario no tiene películas calificadas con 4 o más estrellas."
        )
        return

    # Calcular la similitud entre películas usando la matriz usuario-item transpuesta
    movie_similarity = cosine_similarity(user_item_matrix.T)
    movie_similarity_df = pd.DataFrame(
        movie_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns,
    )

    # Obtener las películas similares a las favoritas
    similar_movies = pd.Series(dtype=float)
    for movie_id in favorite_movies:
        sims = movie_similarity_df[movie_id].drop(index=favorite_movies)
        similar_movies = pd.concat([similar_movies, sims])

    # Agrupar y ordenar las películas similares
    similar_movies = similar_movies.groupby(similar_movies.index).mean()
    similar_movies = similar_movies.sort_values(ascending=False)

    # Eliminar las películas que el usuario ya ha visto
    watched_movies = set(user_ratings["movieId"].unique())
    similar_movies = similar_movies[~similar_movies.index.isin(watched_movies)]

    # Seleccionar las 5 mejores recomendaciones
    recommended_movies = similar_movies.head(5).index

    # Obtener los títulos de las películas recomendadas
    movie_titles = ratings[["movieId", "title"]].drop_duplicates()
    recommended_titles = movie_titles[movie_titles["movieId"].isin(recommended_movies)]

    # Preparar el prompt para la API de OpenAI
    recommended_movie_titles = ", ".join(recommended_titles["title"].tolist())
    prompt = (
        f"El usuario ha disfrutado de las películas: {', '.join(favorite_movie_titles)}. "
        f"Recomiéndale las siguientes películas: {recommended_movie_titles}. "
        "Por favor, explica brevemente por qué estas películas podrían gustarle al usuario."
    )

    try:
        # Hacer una sola llamada a la API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un experto en películas y recomendaciones cinematográficas.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
        )
        description = response.choices[0].message.content.strip()

        # Mostrar las recomendaciones en el sidebar
        st.sidebar.write("**Recomendaciones:**")
        st.sidebar.write(description)
    except Exception as e:
        st.sidebar.error(f"Error al conectar con la API de OpenAI: {e}")


def display_insights(ratings):
    """Muestra análisis e insights del dataset."""
    st.header("MovieLens insights")

    # Películas con más calificaciones
    st.subheader("Top 10 Películas con más Calificaciones")
    top_movies = ratings["title"].value_counts().head(10)
    top_movies_df = top_movies.reset_index()
    top_movies_df.index = top_movies_df.index + 1
    top_movies_df.columns = ["Película", "Número de Calificaciones"]
    st.table(top_movies_df)

    # Distribución de calificaciones
    st.subheader("Distribución de Calificaciones")
    rating_counts = ratings["rating"].value_counts().sort_index()
    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        labels={"x": "Calificación", "y": "Frecuencia"},
    )
    fig.update_layout(
        height=400,
        xaxis_title="Calificación",
        yaxis_title="Número de Calificaciones",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Usuarios con más calificaciones
    st.subheader("Top 10 Usuarios con más Calificaciones")

    top_users = ratings["userId"].value_counts().head(10)
    fig = px.bar(
        x=top_users.index.astype(str),
        y=top_users.values,
        labels={"x": "Usuario", "y": "Número de Calificaciones"},
    )
    fig.update_layout(
        height=400,
        xaxis=dict(
            title="Usuario ID",
            type="category",
        ),
        yaxis_title="Número de Calificaciones",
    )
    st.plotly_chart(fig, use_container_width=True)


def get_top_similar_users(user_item_matrix, selected_user):
    """Obtiene los 10 usuarios más similares al usuario seleccionado."""
    # Calcular la similitud de coseno entre los usuarios
    user_similarity = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(
        user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index
    )

    # Ordenar los usuarios más similares al seleccionado
    top_users = similarity_df[selected_user].sort_values(ascending=False).iloc[1:11]

    # Crear DataFrame para mostrar
    top_users_df = pd.DataFrame(
        {"Usuario": top_users.index, "Similitud": top_users.values}
    ).reset_index(drop=True)

    return top_users_df


if __name__ == "__main__":
    main()
