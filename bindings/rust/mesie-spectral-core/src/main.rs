use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{self, Read};

#[derive(Debug, Deserialize)]
struct Request {
    action: String,
    record: Option<Value>,
    record_a: Option<Value>,
    record_b: Option<Value>,
}

#[derive(Debug, Serialize)]
struct Response {
    ok: bool,
    data: Value,
    vector: Option<Vec<f64>>,
    error: Option<String>,
}

fn primary_arrays(record: &Value) -> (Vec<f64>, Vec<f64>) {
    if let Some(comps) = record.get("components").and_then(|c| c.as_array()) {
        if let Some(c0) = comps.first() {
            let f = c0["frequency"].as_array().unwrap_or(&vec![]);
            let a = c0["amplitude"].as_array().unwrap_or(&vec![]);
            return (
                f.iter().filter_map(|x| x.as_f64()).collect(),
                a.iter().filter_map(|x| x.as_f64()).collect(),
            );
        }
    }
    let f = record["frequency"].as_array().unwrap_or(&vec![]);
    let a = record["amplitude"].as_array().unwrap_or(&vec![]);
    (
        f.iter().filter_map(|x| x.as_f64()).collect(),
        a.iter().filter_map(|x| x.as_f64()).collect(),
    )
}

fn validate(record: &Value) -> Value {
    let (freq, amp) = primary_arrays(record);
    let mut errors = vec![];
    if freq.is_empty() || amp.is_empty() {
        errors.push("missing frequency/amplitude");
    }
    if freq.len() != amp.len() {
        errors.push("length mismatch");
    }
    if amp.iter().any(|x| *x < 0.0) {
        errors.push("negative amplitudes");
    }
    serde_json::json!({
        "is_valid": errors.is_empty(),
        "level": if errors.is_empty() { 5 } else { 2 },
        "errors": errors,
        "warnings": [],
        "runtime": "rust"
    })
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let dot: f64 = a[..n].iter().zip(&b[..n]).map(|(x, y)| x * y).sum();
    let na: f64 = a[..n].iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b[..n].iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot / (na * nb)
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 1.0;
    }
    let mse: f64 = a[..n]
        .iter()
        .zip(&b[..n])
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        / n as f64;
    mse.sqrt()
}

fn match_records(a: &Value, b: &Value) -> Value {
    let (fa, aa) = primary_arrays(a);
    let (fb, ab) = primary_arrays(b);
    let c = cosine(&aa, &ab);
    let r = rmse(&aa, &ab);
    let score = (0.6 * c + 0.4 * (1.0 / (1.0 + r))).clamp(0.0, 1.0);
    serde_json::json!({
        "composite_score": score,
        "metrics": { "cosine": c, "rmse": r },
        "runtime": "rust"
    })
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let req: Request = serde_json::from_str(&input).unwrap_or(Request {
        action: "health".into(),
        record: None,
        record_a: None,
        record_b: None,
    });

    let (ok, data, error) = match req.action.as_str() {
        "health" => (true, serde_json::json!({"status": "ok"}), None),
        "validate" => {
            let rec = req.record.unwrap_or(Value::Null);
            (true, validate(&rec), None)
        }
        "match" => {
            let a = req.record_a.or(req.record).unwrap_or(Value::Null);
            let b = req.record_b.unwrap_or(Value::Null);
            (true, match_records(&a, &b), None)
        }
        other => (
            false,
            Value::Null,
            Some(format!("unsupported action: {}", other)),
        ),
    };

    let resp = Response {
        ok,
        data,
        vector: None,
        error,
    };
    println!("{}", serde_json::to_string(&resp).unwrap());
}