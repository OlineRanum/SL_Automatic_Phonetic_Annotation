// routes/dataMocapRoutes.js
const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path'); // Import the 'path' module
const dataMocapController = require('../controllers/dataMocapController');

// Configure multer
const upload = multer({
    dest: path.join(__dirname, '..', '..','data',  'mocap'), // Adjusted the path to navigate correctly
    limits: { fileSize: 50 * 1024 * 1024 }, // 50 MB limit
});

// Upload multiple files
router.post('/', upload.array('mocapFile', 10), dataMocapController.uploadFiles);

// List uploaded files
router.get('/', dataMocapController.listFiles);

// Delete files
router.delete('/', dataMocapController.deleteFiles);

module.exports = router;
