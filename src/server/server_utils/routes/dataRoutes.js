// server_utils/routes/dataRoutes.js
const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const dataController = require('../controllers/dataController');

// We’ll store files initially in a temporary folder, then move them
// to the correct subfolder (mocap, video, etc.) inside the controller.
const upload = multer({
  dest: path.join(__dirname, '..', '..', 'public', 'data', 'temp'),
  limits: { fileSize: 50 * 1024 * 1024 } // 50 MB limit
});

// 1) Upload multiple files, using “:type” to decide which subfolder to use
//    e.g., POST /
// or /api/data/video
router.post('/:type', upload.array('files', 10), dataController.uploadFiles);

// 2) List files in a given subfolder
//    e.g., GET /api/data/mocap or GET /api/data/video
router.get('/:type', dataController.listFiles);

// 3) Delete files from a given subfolder
//    e.g., DELETE /api/data/mocap or DELETE /api/data/video
router.delete('/:type', dataController.deleteFiles);

module.exports = router;

