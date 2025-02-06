// routes/notesRoutes.js
const express = require('express');
const router = express.Router();
const path = require('path'); // Import the 'path' module
const notesController = require('../controllers/notesController');

// Get all notes
router.get('/', notesController.getNotes);

// Get notes for a specific frame
router.get('/:gifName/frames/:frameName/notes', notesController.getFrameNotes);

// Add a note to a specific frame
router.post('/:gifName/frames/:frameName/notes', notesController.addNote);

// Delete a specific note
router.delete('/:gifName/frames/:frameName/notes/:noteId', notesController.deleteNote);

module.exports = router;
