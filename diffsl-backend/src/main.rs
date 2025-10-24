use axum::{
    body::Body, extract, http::{header, HeaderMap, StatusCode}, response::{IntoResponse, Response}, routing::{get, post}, Router
};
use diffsl::{execution::compiler::CompilerMode, Compiler, LlvmModule};
use hyper::Method;
use serde::{Deserialize, Serialize};
use std::{env::temp_dir, net::SocketAddr, path::Path};
use tokio_util::io::ReaderStream;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

async fn hello_world() -> &'static str {
    "Hello, World!"
}

//async fn compile(body: String) -> Result<Response, AppError> {
//    let filepath = temp_dir().join("model.wasm");
//    let filename = filepath.into_os_string().into_string().unwrap();
//
//    {
//        let compiler = Compiler::<LlvmModule>::from_discrete_str(payload.text.as_str(), CompilerMode::SingleThreaded)?;
//        compiler.module().compile(true, true, filename.as_str())?;
//    }
//
//    let file = tokio::fs::File::open(Path::new(filename.as_str())).await?;
//
//    let stream = ReaderStream::new(file);
//
//    let body = Body::from_stream(stream);
//
//    let mut headers = HeaderMap::new();
//    headers.insert(header::CONTENT_TYPE, "application/wasm".parse().unwrap());
//
//    Ok((headers, body).into_response())
//}

fn app() -> Router {
    let cors = CorsLayer::new()
        // allow `GET` and `POST` when accessing the resource
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        // allow requests from any origin
        .allow_origin(Any)
        .allow_headers(vec![header::ACCEPT, header::CONTENT_TYPE]);
    Router::new()
        .route("/", get(hello_world))
        //.route("/compile", post(compile))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
}


#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let app = app();
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    let listener = tokio::net::TcpListener::bind(addr.to_string())
        .await
        .unwrap();
    println!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

// Make our own error that wraps `anyhow::Error`.
struct AppError(anyhow::Error);

// Tell axum how to convert `AppError` into a response.
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (StatusCode::INTERNAL_SERVER_ERROR, self.0.to_string()).into_response()
    }
}

// This enables using `?` on functions that return `Result<_, anyhow::Error>` to turn them into
// `Result<_, AppError>`. That way you don't need to do that manually.
impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}