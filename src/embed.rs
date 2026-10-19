use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::config::EmbeddingConfig;

/// Client for an OpenAI-compatible `/v1/embeddings` endpoint.
pub struct Embedder<'a> {
    config: &'a EmbeddingConfig,
    agent: ureq::Agent,
}

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    input: &'a [String],
    encoding_format: &'static str,
}

#[derive(Deserialize)]
struct Response {
    data: Vec<Item>,
}

#[derive(Deserialize)]
struct Item {
    index: usize,
    embedding: Vec<f32>,
}

impl<'a> Embedder<'a> {
    pub fn new(config: &'a EmbeddingConfig) -> Self {
        let timeout = Duration::from_secs(config.timeout_secs.max(1));
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(timeout))
            .timeout_connect(Some(Duration::from_secs(10).min(timeout)))
            .build()
            .into();
        Self { config, agent }
    }

    /// Returns one unit-length vector per input, in input order.
    pub fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = &self.config.url;
        let mut request = self.agent.post(url);
        if let Some(key) = &self.config.api_key {
            request = request.header("Authorization", &format!("Bearer {key}"));
        }
        let mut response = request
            .send_json(Request {
                model: &self.config.model,
                input: inputs,
                encoding_format: "float",
            })
            .with_context(|| {
                format!(
                    "is the embedding server running at {url}? (gave up after {}s)",
                    self.config.timeout_secs.max(1)
                )
            })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.body_mut().read_to_string().unwrap_or_default();
            bail!("embedding server returned {status}: {}", body.trim());
        }
        let mut data = response
            .body_mut()
            .read_json::<Response>()
            .context("parsing embedding response")?
            .data;
        ensure!(
            data.len() == inputs.len(),
            "embedding server returned {} embeddings for {} inputs",
            data.len(),
            inputs.len()
        );
        data.sort_by_key(|item| item.index);
        ensure!(
            data.iter().enumerate().all(|(i, item)| item.index == i),
            "embedding server returned duplicate or out-of-range indices"
        );

        let dimensions = self.config.dimensions;
        data.into_iter()
            .map(|item| enforce_dimensions(item.embedding, dimensions))
            .collect()
    }
}

/// Truncates to `dimensions` (Matryoshka) and returns the vector at unit length.
fn enforce_dimensions(mut v: Vec<f32>, dimensions: usize) -> Result<Vec<f32>> {
    ensure!(
        v.len() >= dimensions,
        "embedding server returned {}-dimensional vectors, but embedding.dimensions is {dimensions}",
        v.len()
    );
    v.truncate(dimensions);
    Ok(normalized(v))
}

/// Unit length makes cosine similarity a plain dot product at search time.
fn normalized(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_then_normalizes() {
        assert_eq!(
            enforce_dimensions(vec![3.0, 4.0, 100.0], 2).unwrap(),
            vec![0.6, 0.8]
        );
    }

    #[test]
    fn keeps_exact_length() {
        assert_eq!(
            enforce_dimensions(vec![3.0, 4.0], 2).unwrap(),
            vec![0.6, 0.8]
        );
    }

    #[test]
    fn rejects_shorter() {
        assert!(enforce_dimensions(vec![1.0], 2).is_err());
    }
}
