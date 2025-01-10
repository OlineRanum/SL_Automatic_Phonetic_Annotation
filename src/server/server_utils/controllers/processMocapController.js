// controllers/processMocapController.js

const path = require('path');
const { spawn } = require('child_process');
const ProcessingState = require('../utils/ProcessingState'); // Singleton instance
const fs = require('fs'); // To check file existence

exports.processMocapFiles = (req, res, progressData) => {
    progressData.progress = 0;
    const { files } = req.body;

    console.log('Received processMocapFiles request');
    console.log(`Files received: ${JSON.stringify(files)}`);

    if (!files || !Array.isArray(files) || files.length === 0) {
        console.log('No files selected for processing.');
        return res.status(400).json({ status: 'error', message: 'No files selected for processing.' });
    }

    if (ProcessingState.isProcessing) {
        console.log('Processing already in progress.');
        return res.status(400).json({ status: 'error', message: 'Processing already in progress.' });
    }

    ProcessingState.isProcessing = true;
    console.log(`Processing started for files: ${files.join(', ')}`);

    try {
        const scriptPath = path.resolve(__dirname, '../../../modules/data/mocap_data/visualize_reference_data.py');
        console.log(`Resolved Python script path: ${scriptPath}`);

        // Check if the Python script exists
        if (!fs.existsSync(scriptPath)) {
            console.error(`Python script not found at path: ${scriptPath}`);
            ProcessingState.isProcessing = false;
            return res.status(500).json({ status: 'error', message: 'Processing script not found on server.' });
        }

        // Optional: Log each file's expected path
        files.forEach(file => {
            const filePath = path.resolve(__dirname, '../../../modules/data/mocap_data/', file);
            console.log(`Expected file path: ${filePath}`);
            if (!fs.existsSync(filePath)) {
                console.warn(`File does not exist: ${filePath}`);
                // Optionally, handle missing files here
            }
        });

        const pythonProcess = spawn('python3', [scriptPath, '--data_list', ...files]);

        pythonProcess.stdout.on('data', (data) => {
            const message = data.toString();
            console.log(`Python output: ${message}`);

            if (message.startsWith('Processing file')) {
                progressData.currentFile = message.split(': ')[1];
                progressData.progress = 0;
            } else if (message.startsWith('Progress:')) {
                progressData.progress = parseInt(message.split(': ')[1], 10);
            }
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error(`Python error: ${data}`);
        });

        pythonProcess.on('close', (code) => {
            console.log(`Python script finished with code ${code}`);
            ProcessingState.isProcessing = false;
            progressData.message = 'Processing complete!';

            if (code !== 0) {
                console.error(`Python script exited with non-zero code: ${code}`);
                // Optionally, send an error notification to the client
            }
        });

        pythonProcess.on('error', (err) => {
            console.error(`Python process error: ${err}`);
            ProcessingState.isProcessing = false;
        });

        // Handle unexpected errors
        process.on('uncaughtException', (err) => {
            console.error('Unexpected error:', err);
            ProcessingState.isProcessing = false;
        });

        // Send initial response to the client
        res.status(200).json({ status: 'success', message: 'Processing started.' });
    } catch (err) {
        console.error(`Server error: ${err.message}`);
        ProcessingState.isProcessing = false;
        res.status(500).json({ status: 'error', message: 'Internal server error.' });
    }
};
