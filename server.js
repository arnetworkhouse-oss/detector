const express = require("express");
const multer = require("multer");
const cors = require("cors");

const app = express();
app.use(cors());

const upload = multer({ dest: "uploads/" });

app.get("/", (req, res) => {
  res.send("API is running");
});

app.post("/detect", upload.single("file"), (req, res) => {
  // Fake response for now (testing)
  res.json({
    success: true,
    ai_probability: Math.floor(Math.random() * 100)
  });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log("Server running on port " + PORT));
