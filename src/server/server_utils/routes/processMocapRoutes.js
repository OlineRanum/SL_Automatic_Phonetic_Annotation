// routes/processMocapRoutes.js

const express = require('express');
const router = express.Router();
const { processMocapFiles } = require('../controllers/processMocapController');
const ProcessingState = require('../utils/ProcessingState'); // Singleton instance

// Store progress updates (e.g., in-memory object)
let progressData = {};

// POST Route: Start processing
router.post('/', (req, res) => {
    processMocapFiles(req, res, progressData);
});

// GET Route: Progress updates
router.get('/progress', (req, res) => {
    if (!ProcessingState.isProcessing) {
        return res.status(400).json({ status: 'error', message: 'No processing in progress.' });
    }

    // Proceed with SSE connection logic
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const sendProgress = () => {
        const data = JSON.stringify(progressData);
        res.write(`data: ${data}\n\n`);
    };

    const intervalId = setInterval(() => {
        sendProgress();
    }, 1000);

    req.on('close', () => {
        clearInterval(intervalId);
        console.log('Client disconnected from progress updates.');
        progressData.progress = 0; // Reset progress
        progressData.message = ''; // Clear message
        progressData.currentFile = '';
    });
});

module.exports = router;
