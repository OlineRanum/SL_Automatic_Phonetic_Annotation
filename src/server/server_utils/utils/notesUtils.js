// utils/notesUtils.js
const path = require('path');
const { readJSON, writeJSON } = require('./fileUtils');

const notesFilePath = path.join(__dirname, '..','..','public', 'output', 'notes.json');

function readNotes() {
    return readJSON(notesFilePath, {});
}

function writeNotes(notes) {
    writeJSON(notesFilePath, notes);
}

module.exports = {
    readNotes,
    writeNotes,
};
