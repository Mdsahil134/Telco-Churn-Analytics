import cors from "cors";
import express from "express";
import fs from "fs";
import { MongoClient } from "mongodb";
import { loadEnvFile } from "node:process";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ENV_PATH = path.resolve(__dirname, ".env");

if (fs.existsSync(ENV_PATH)) {
  loadEnvFile(ENV_PATH);
}

const PORT = Number(process.env.PORT || 4000);
const SUMMARY_PATH = path.resolve(
  __dirname,
  "..",
  "churn-basic",
  "evaluation_summary.json"
);
const FEATURE_PLOT_PATH = path.resolve(
  __dirname,
  "..",
  "churn-basic",
  "feature_importance.png"
);

const app = express();
app.use(cors());
app.use(express.json());

let client;
let db;

function loadSummary() {
  if (!fs.existsSync(SUMMARY_PATH)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(SUMMARY_PATH, "utf8"));
}

app.get("/api/health", async (_req, res) => {
  try {
    await db.command({ ping: 1 });
    res.json({ ok: true, mongo: "connected" });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.get("/api/dashboard", async (_req, res) => {
  try {
    const customers = db.collection("customers");
    const predictions = db.collection("predictions");

    const totalCustomers = await customers.countDocuments();
    const churnedCount = await customers.countDocuments({ Churn: "Yes" });
    const highRiskCount = await predictions.countDocuments({ risk_level: "High" });

    const avgResult = await predictions
      .aggregate([
        {
          $group: {
            _id: null,
            averageProbability: { $avg: "$churn_probability" },
            scoredAt: { $max: "$scored_at" },
            modelUsed: { $first: "$model_used" },
          },
        },
      ])
      .toArray();

    const contractRows = await customers
      .aggregate([
        {
          $group: {
            _id: "$Contract",
            total: { $sum: 1 },
            churned: {
              $sum: { $cond: [{ $eq: ["$Churn", "Yes"] }, 1, 0] },
            },
          },
        },
      ])
      .toArray();

    const riskRows = await predictions
      .aggregate([
        { $group: { _id: "$risk_level", count: { $sum: 1 } } },
      ])
      .toArray();

    const contractOrder = ["Month-to-month", "One year", "Two year"];
    const churnByContract = contractOrder
      .map((name) => {
        const row = contractRows.find((item) => item._id === name) || {
          total: 0,
          churned: 0,
        };
        return {
          contract: name,
          customers: row.total,
          churned: row.churned,
          churnRate: row.total ? row.churned / row.total : 0,
        };
      })
      .concat(
        contractRows
          .filter((item) => !contractOrder.includes(item._id))
          .map((row) => ({
            contract: row._id || "Unknown",
            customers: row.total,
            churned: row.churned,
            churnRate: row.total ? row.churned / row.total : 0,
          }))
      );

    res.json({
      totalCustomers,
      churnedCount,
      retainedCount: totalCustomers - churnedCount,
      churnRate: totalCustomers ? churnedCount / totalCustomers : 0,
      highRiskCount,
      averageProbability: avgResult[0]?.averageProbability ?? 0,
      modelUsed: avgResult[0]?.modelUsed ?? null,
      scoredAt: avgResult[0]?.scoredAt ?? null,
      churnByContract,
      riskCounts: {
        High: riskRows.find((row) => row._id === "High")?.count ?? 0,
        Medium: riskRows.find((row) => row._id === "Medium")?.count ?? 0,
        Low: riskRows.find((row) => row._id === "Low")?.count ?? 0,
      },
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/customers", async (req, res) => {
  try {
    const contract = String(req.query.contract || "").trim();
    const churn = String(req.query.churn || "").trim();
    const tenureMin = req.query.tenureMin === undefined ? null : Number(req.query.tenureMin);
    const tenureMax = req.query.tenureMax === undefined ? null : Number(req.query.tenureMax);
    const page = Math.max(1, Number(req.query.page || 1));
    const pageSize = Math.min(50, Math.max(5, Number(req.query.pageSize || 12)));

    const match = {};
    if (contract) match.Contract = contract;
    if (churn === "Yes" || churn === "No") match.Churn = churn;
    if (Number.isFinite(tenureMin) || Number.isFinite(tenureMax)) {
      match.tenure = {};
      if (Number.isFinite(tenureMin)) match.tenure.$gte = tenureMin;
      if (Number.isFinite(tenureMax)) match.tenure.$lte = tenureMax;
    }

    const customers = db.collection("customers");
    const [total, summary, pageRows] = await Promise.all([
      customers.countDocuments(match),
      customers
        .aggregate([
          { $match: match },
          {
            $group: {
              _id: "$Churn",
              count: { $sum: 1 },
              avgTenure: { $avg: "$tenure" },
              avgMonthly: { $avg: "$MonthlyCharges" },
              avgTotal: {
                $avg: {
                  $convert: {
                    input: "$TotalCharges",
                    to: "double",
                    onError: 0,
                    onNull: 0,
                  },
                },
              },
            },
          },
        ])
        .toArray(),
      customers
        .find(match, {
          projection: {
            _id: 0,
            customerID: 1,
            gender: 1,
            SeniorCitizen: 1,
            Partner: 1,
            Dependents: 1,
            tenure: 1,
            Contract: 1,
            InternetService: 1,
            PaymentMethod: 1,
            MonthlyCharges: 1,
            TotalCharges: 1,
            Churn: 1,
          },
        })
        .sort({ customerID: 1 })
        .skip((page - 1) * pageSize)
        .limit(pageSize)
        .toArray(),
    ]);

    const ids = pageRows.map((row) => row.customerID);
    const scores = ids.length
      ? await db
          .collection("predictions")
          .find(
            { customerID: { $in: ids } },
            { projection: { _id: 0, customerID: 1, churn_probability: 1, risk_level: 1 } }
          )
          .toArray()
      : [];
    const scoreById = Object.fromEntries(scores.map((item) => [item.customerID, item]));
    const rows = pageRows.map((row) => ({
      ...row,
      churn_probability: scoreById[row.customerID]?.churn_probability ?? null,
      risk_level: scoreById[row.customerID]?.risk_level ?? null,
    }));

    const comparison = { churned: null, retained: null };
    for (const row of summary) {
      const payload = {
        count: row.count,
        avgTenure: row.avgTenure ?? 0,
        avgMonthly: row.avgMonthly ?? 0,
        avgTotal: row.avgTotal ?? 0,
      };
      if (row._id === "Yes") comparison.churned = payload;
      if (row._id === "No") comparison.retained = payload;
    }

    res.json({
      page,
      pageSize,
      total,
      totalPages: Math.max(1, Math.ceil(total / pageSize)),
      comparison,
      rows,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/predictions", async (req, res) => {
  try {
    const customerID = String(req.query.customerID || "").trim();
    const risk = String(req.query.risk || "").trim();
    const sortDir = String(req.query.sort || "desc").toLowerCase() === "asc" ? 1 : -1;
    const page = Math.max(1, Number(req.query.page || 1));
    const pageSize = Math.min(50, Math.max(5, Number(req.query.pageSize || 12)));

    const match = {};
    if (customerID) {
      match.customerID = { $regex: customerID, $options: "i" };
    }
    if (["High", "Medium", "Low"].includes(risk)) {
      match.risk_level = risk;
    }

    const collection = db.collection("predictions");
    const total = await collection.countDocuments(match);
    const rows = await collection
      .find(match, { projection: { _id: 0 } })
      .sort({ churn_probability: sortDir, customerID: 1 })
      .skip((page - 1) * pageSize)
      .limit(pageSize)
      .toArray();

    res.json({
      page,
      pageSize,
      total,
      totalPages: Math.max(1, Math.ceil(total / pageSize)),
      rows,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/model", (_req, res) => {
  const summary = loadSummary();
  if (!summary) {
    return res.status(404).json({
      error: "evaluation_summary.json was not found. Run python train.py first.",
    });
  }
  res.json(summary);
});

app.get("/api/feature-importance.png", (_req, res) => {
  if (!fs.existsSync(FEATURE_PLOT_PATH)) {
    return res.status(404).json({ error: "feature_importance.png is missing." });
  }
  res.sendFile(FEATURE_PLOT_PATH);
});

async function start() {
  if (!process.env.MONGODB_URI) {
    throw new Error("MONGODB_URI environment variable is missing");
  }

  client = new MongoClient(process.env.MONGODB_URI, {
    serverSelectionTimeoutMS: 20000,
    connectTimeoutMS: 20000,
  });

  await client.connect();
  console.log("MongoDB connected successfully");

  db = client.db("churn_db");
  await db.command({ ping: 1 });

  // Auto-sync predictions from predictions_seed.json if remote collection has outdated predictions (e.g. 2,223 high risk from 10/16 run)
  const SEED_FILE = path.resolve(__dirname, "predictions_seed.json");
  if (fs.existsSync(SEED_FILE)) {
    try {
      const highRiskInDb = await db.collection("predictions").countDocuments({ risk_level: "High" });
      const totalInDb = await db.collection("predictions").countDocuments({});
      if (totalInDb === 0 || highRiskInDb > 1000) {
        console.log(`Outdated or missing predictions detected (${highRiskInDb} high risk, ${totalInDb} total). Syncing from predictions_seed.json...`);
        const seedData = JSON.parse(fs.readFileSync(SEED_FILE, "utf8"));
        const documents = seedData.map((d) => ({
          ...d,
          scored_at: d.scored_at ? new Date(d.scored_at) : new Date(),
        }));
        await db.collection("predictions").drop().catch(() => {});
        const res = await db.collection("predictions").insertMany(documents);
        console.log(`Successfully synced ${res.insertedCount} updated predictions into churn_db.predictions`);
      }
    } catch (seedErr) {
      console.warn("Prediction seed sync warning:", seedErr.message);
    }
  }

  await db.collection("customers").createIndex({ customerID: 1 });
  await db.collection("predictions").createIndex({ customerID: 1 });
  await db.collection("predictions").createIndex({ churn_probability: -1 });
  await db.collection("predictions").createIndex({ risk_level: 1 });
  app.listen(PORT, () => {
    console.log(`Dashboard API listening on http://localhost:${PORT}`);
  });
}

start().catch((error) => {
  console.error("Failed to start dashboard API:", error.message);
  process.exit(1);
});
