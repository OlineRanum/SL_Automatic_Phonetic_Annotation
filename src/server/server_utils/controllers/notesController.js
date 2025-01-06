// controllers/notesController.js
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const { readNotes, writeNotes } = require('../utils/notesUtils');

exports.getNotes = (req, res) => {
    const notes = readNotes();
    res.json(notes);
};

exports.getFrameNotes = (req, res) => {
    const { gifName, frameName } = req.params;
    const notes = readNotes();

    if (notes[gifName] && notes[gifName][frameName]) {
        res.json(notes[gifName][frameName]);
    } else {
        res.json([]);
    }
};

exports.addNote = (req, res) => {
    const { gifName, frameName } = req.params;
    const { content } = req.body;

    if (!content || typeof content !== 'string') {
        console.warn('Invalid note content received.');
        return res.status(400).json({ error: 'Invalid note content.' });
    }

    const notes = readNotes();

    if (!notes[gifName]) {
        notes[gifName] = {};
    }

    if (!notes[gifName][frameName]) {
        notes[gifName][frameName] = [];
    }

    const newNote = {
        id: uuidv4(),
        timestamp: new Date().toISOString(),
        content: content.trim(),
    };

    notes[gifName][frameName].push(newNote);
    writeNotes(notes);

    console.log(`Added new note to ${gifName} - ${frameName}:`, newNote);
    res.json(newNote);
};

exports.deleteNote = (req, res) => {
    const { gifName, frameName, noteId } = req.params;
    const notes = readNotes();

    if (
        notes[gifName] &&
        notes[gifName][frameName] &&
        Array.isArray(notes[gifName][frameName])
    ) {
        const noteIndex = notes[gifName][frameName].findIndex(note => note.id === noteId);
        if (noteIndex !== -1) {
            const deletedNote = notes[gifName][frameName].splice(noteIndex, 1)[0];
            writeNotes(notes);
            console.log(`Deleted note from ${gifName} - ${frameName}:`, deletedNote);
            return res.json({ message: 'Note deleted successfully.', deletedNote });
        } else {
            console.warn(`Note ID ${noteId} not found in ${gifName} - ${frameName}.`);
            return res.status(404).json({ error: 'Note not found.' });
        }
    } else {
        console.warn(`No notes found for ${gifName} - ${frameName}.`);
        return res.status(404).json({ error: 'Note not found.' });
    }
};
