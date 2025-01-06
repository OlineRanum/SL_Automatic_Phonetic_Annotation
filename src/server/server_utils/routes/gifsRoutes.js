// routes/gifsRoutes.js
const express = require('express');
const router = express.Router();
const gifsController = require('../controllers/gifsController');
router.get('/', gifsController.listGifs);
router.get('/:gifName/frames', gifsController.getGifFrames);

module.exports = router;

