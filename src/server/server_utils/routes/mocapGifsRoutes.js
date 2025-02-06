// routes/mocapGifsRoutes.js
const express = require('express');
const router = express.Router();
const mocapGifsController = require('../controllers/mocapGifsController');

// List all MoCap GIFs
router.get('/', mocapGifsController.listMocapGifs);

// Get frames for a specific MoCap GIF
router.get('/:gifName/frames', mocapGifsController.getMocapGifFrames);

// Get selected frames for a specific MoCap GIF
router.get('/:gifName/selected_frames', mocapGifsController.getSelectedFrames);

// Add selected frames for a specific MoCap GIF
router.post('/:gifName/selected_frames', mocapGifsController.addSelectedFrames);

// Delete selected frames for a specific MoCap GIF
router.delete('/:gifName/selected_frames', mocapGifsController.deleteSelectedFrames);

module.exports = router;
