// utils/fileUtils.js
const fs = require('fs');
const path = require('path');

// Generic function to read JSON files
function readJSON(filePath, defaultData = {}) {
    try {
        if (!fs.existsSync(filePath)) {
            fs.writeFileSync(filePath, JSON.stringify(defaultData, null, 2), 'utf8');
            console.log(`Created new ${path.basename(filePath)} file.`);
            return defaultData;
        }
        const data = fs.readFileSync(filePath, 'utf8');
        return JSON.parse(data);
    } catch (err) {
        console.error(`Error reading ${path.basename(filePath)}:`, err);
        return defaultData;
    }
}

// Generic function to write JSON files
function writeJSON(filePath, data) {
    try {
        fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf8');
        console.log(`Successfully wrote to ${path.basename(filePath)}.`);
    } catch (err) {
        console.error(`Error writing to ${path.basename(filePath)}:`, err);
    }
}

module.exports = {
    readJSON,
    writeJSON,
};
