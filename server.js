const express = require("express");
const multer = require("multer");
const cors = require("cors");

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// File upload config
const upload = multer({ dest: "uploads/" });


// ✅ Home route (to fix "Not Found")
app.get("/", (req, res) => {
  res.send("API is running 🚀");
});


// ✅ Detect route (POST)
app.post("/detect", upload.single("file"), (req, res) => {
  try {
    // Check if file exists
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: "No file uploaded"
      });
    }

    // Fake AI response (for now)
    const probability = Math.floor(Math.random() * 100);

    res.json({
      success: true,
      filename: req.file.originalname,
      ai_probability: probability
    });

  } catch (error) {
    console.error(error);
    res.status(500).json({
      success: false,
      message: "Server error"
    });
  }
});


// ✅ Start server
const PORT = process.env.PORT || 10000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
