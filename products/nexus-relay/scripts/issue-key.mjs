import crypto from "node:crypto";

const [email = "customer@example.com", plan = "developer", monthly = "1000", rpm = "30"] = process.argv.slice(2);
const customerId = `cus_${crypto.randomBytes(8).toString("hex")}`;
const keyId = `key_${crypto.randomBytes(8).toString("hex")}`;
const apiKey = `nr_live_${crypto.randomBytes(24).toString("base64url")}`;
const keyHash = crypto.createHash("sha256").update(apiKey).digest("hex");
const now = new Date().toISOString();
const esc = (value) => String(value).replaceAll("'", "''");

console.log(JSON.stringify({ customer_id: customerId, key_id: keyId, api_key: apiKey, plan, monthly_limit: Number(monthly), rate_limit_per_minute: Number(rpm) }, null, 2));
console.error("\nRun this SQL against the Relay D1 database. Store the plaintext API key only in your secure customer-delivery system.\n");
console.error(`INSERT INTO customers (id,email,name,status,created_at) VALUES ('${customerId}','${esc(email)}','${esc(email)}','active','${now}');`);
console.error(`INSERT INTO api_keys (id,customer_id,key_hash,name,plan,active,monthly_limit,rate_limit_per_minute,created_at) VALUES ('${keyId}','${customerId}','${keyHash}','default','${esc(plan)}',1,${Number(monthly)},${Number(rpm)},'${now}');`);
