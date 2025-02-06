// server_utils/utils/config.js

const path = require('path');

// Define the project root based on the current file's location
const projectRoot = path.resolve(__dirname, '..', '..');

module.exports = {
    projectRoot,
    publicDir: path.join(projectRoot, 'public'),
    uploadsDir: path.join(projectRoot, 'uploads'),
    selectedFramesFile: path.join(projectRoot, 'selected_frames.json'),
    notesFilePath: path.join(projectRoot, 'public', 'output', 'notes.json'),
    // Add other paths as needed
};
