use anyhow::anyhow;
use axum::{
    body::Body,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use diffsl::{
    discretise::DiscreteModel, execution::compiler::CompilerMode, parser::parse_ds_string,
    CodegenModuleCompile, CodegenModuleEmit, LlvmModule,
};
use hyper::Method;
use std::net::SocketAddr;
use target_lexicon::Triple;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

async fn hello_world() -> &'static str {
    "Hello, World!"
}

async fn compile(body: String) -> Result<Response, AppError> {
    let object = {
        let name = "diffsl";
        let model = parse_ds_string(body.as_str()).map_err(|e| anyhow!(e.to_string()))?;
        let model = DiscreteModel::build(name, &model)
            .map_err(|e| anyhow!(e.as_error_message(body.as_str())))?;

        let mode = CompilerMode::SingleThreaded;
        let triple = Triple {
            architecture: target_lexicon::Architecture::Wasm32,
            vendor: target_lexicon::Vendor::Unknown,
            operating_system: target_lexicon::OperatingSystem::Unknown,
            environment: target_lexicon::Environment::Unknown,
            binary_format: target_lexicon::BinaryFormat::Wasm,
        };
        let module = LlvmModule::from_discrete_model(&model, mode, Some(triple))?;
        module.to_object()?
    };

    let body = Body::from(object);

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, "application/wasm".parse().unwrap());

    Ok((headers, body).into_response())
}

fn app() -> Router {
    let cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_origin(Any)
        .allow_headers(vec![header::ACCEPT, header::CONTENT_TYPE]);
    Router::new()
        .route("/", get(hello_world))
        .route("/compile", post(compile))
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use tokio::io::AsyncWriteExt;
    use tower::ServiceExt;

    #[tokio::test]
    async fn hello() {
        let app = app();

        // `Router` implements `tower::Service<Request<Body>>` so we can
        // call it like any tower service, no need to run an HTTP server.
        let response = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
                .unwrap(),
            "*",
        );

        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"Hello, World!");
    }

    #[tokio::test]
    async fn discrete_logistic_model() {
        let text = String::from(
            "
            in = [r, k]
            r { 1 }
            k { 1 }
            u_i {
                y = 1,
                z = 0,
            }
            dudt_i {
                dydt = 0,
                dzdt = 0,
            }
            M_i {
                dydt,
                0,
            }
            F_i {
                (r * y) * (1 - (y / k)),
                (2 * y) - z,
            }
            out_i {
                y,
                z,
            }
        ",
        );
        let request = Request::builder()
            .uri("/compile")
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(text))
            .unwrap();

        let app = app();
        let response = app.oneshot(request).await.unwrap();

        let status = response.status();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let filename = "model.wasm";
        let mut file = tokio::fs::File::create(filename).await.unwrap();
        file.write_all(&body).await.unwrap();

        if status != StatusCode::OK {
            println!("Error recieved: {:?}", body);
        }

        assert_eq!(status, StatusCode::OK);
    }
}
