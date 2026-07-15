// AISVectorPolyglot Motoko canister — on-chain spectral validation gate
// Deploy with dfx; Python MotokoAdapter mirrors this logic locally.

import Array "mo:base/Array";
import Float "mo:base/Float";
import Text "mo:base/Text";
import Nat "mo:base/Nat";

actor MesieCanister {

  public type Component = {
    name : Text;
    frequency : [Float];
    amplitude : [Float];
  };

  public type SpectralRecord = {
    record_id : Text;
    components : [Component];
  };

  public type ValidationResult = {
    is_valid : Bool;
    level : Nat;
    errors : [Text];
  };

  public type MatchResult = {
    composite_score : Float;
    cosine : Float;
    rmse : Float;
  };

  func dot(a : [Float], b : [Float]) : Float {
    var s : Float = 0.0;
    let n = Nat.min(a.size(), b.size());
    var i : Nat = 0;
    while (i < n) {
      s += a[i] * b[i];
      i += 1;
    };
    s
  };

  public func validate(record : SpectralRecord) : async ValidationResult {
    if (record.components.size() == 0) {
      return { is_valid = false; level = 1; errors = ["no components"] };
    };
    let c = record.components[0];
    if (c.frequency.size() != c.amplitude.size()) {
      return { is_valid = false; level = 2; errors = ["length mismatch"] };
    };
    for (a in c.amplitude.vals()) {
      if (a < 0.0) {
        return { is_valid = false; level = 3; errors = ["negative amplitude"] };
      };
    };
    { is_valid = true; level = 5; errors = [] }
  };

  public func match_records(a : SpectralRecord, b : SpectralRecord) : async MatchResult {
    let aa = a.components[0].amplitude;
    let ab = b.components[0].amplitude;
    let cosine = dot(aa, ab);
    let rmse = 0.1;
    let score = 0.6 * cosine + 0.4 * (1.0 / (1.0 + rmse));
    { composite_score = score; cosine = cosine; rmse = rmse }
  };
};