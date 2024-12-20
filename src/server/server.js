// server.js

const express = require('express');
const path = require('path');
const fs = require('fs');
const bodyParser = require('body-parser');
const app = express();
const { v4: uuidv4 } = require('uuid');

// Middleware to parse JSON bodies
app.use(bodyParser.json());

// Serve static files from the "public" directory
app.use(express.static(path.join(__dirname, 'public')));

// Path to the notes.json file
const notesFilePath = path.join(__dirname, 'notes.json');

// Utility function to read notes from the JSON file
function readNotes() {
    try {
        if (!fs.existsSync(notesFilePath)) {
            // If notes.json doesn't exist, create an empty object
            fs.writeFileSync(notesFilePath, JSON.stringify({}, null, 4), 'utf8');
            console.log('Created new notes.json file.');
            return {};
        }
        const data = fs.readFileSync(notesFilePath, 'utf8');
        return JSON.parse(data);
    } catch (err) {
        console.error('Error reading notes.json:', err);
        return {};
    }
}

// Utility function to write notes to the JSON file
function writeNotes(notes) {
    try {
        fs.writeFileSync(notesFilePath, JSON.stringify(notes, null, 4), 'utf8');
        console.log('Successfully wrote to notes.json.');
    } catch (err) {
        console.error('Error writing to notes.json:', err);
    }
}

// API endpoint to list all GIFs
app.get('/api/gifs', (req, res) => {
    const gifsDir = path.join(__dirname, 'public', 'gifs');
    fs.readdir(gifsDir, (err, files) => {
        if (err) {
            console.error('Error reading GIFs directory:', err);
            return res.status(500).json({ error: 'Unable to scan GIFs directory.' });
        }
        // Filter only GIF files
        const gifFiles = files.filter(file => path.extname(file).toLowerCase() === '.gif');
        res.json(gifFiles);
    });
});

// API endpoint to list all Reference Pose PNGs
app.get('/api/reference_poses', (req, res) => {
    const refPosesDir = path.join(__dirname, 'public', 'reference_poses');
    fs.readdir(refPosesDir, (err, files) => {
        if (err) {
            console.error('Error reading Reference Poses directory:', err);
            return res.status(500).json({ error: 'Unable to scan Reference Poses directory.' });
        }
        // Filter only PNG files
        const pngFiles = files.filter(file => path.extname(file).toLowerCase() === '.png');
        res.json(pngFiles);
    });
});

// API endpoint to get frames for a specific GIF
app.get('/api/gifs/:gifName/frames', (req, res) => {
    const gifName = req.params.gifName;
    const framesDir = path.join(__dirname, 'public', 'frames', path.parse(gifName).name);

    // Check if frames directory exists
    if (!fs.existsSync(framesDir)) {
        console.warn(`Frames directory not found for GIF: ${gifName}`);
        return res.status(404).json({ error: 'Frames not found for the selected GIF.' });
    }

    // Read frame files
    fs.readdir(framesDir, (err, files) => {
        if (err) {
            console.error('Error reading frames directory:', err);
            return res.status(500).json({ error: 'Unable to read frames directory.' });
        }

        // Sort frames numerically based on filename
        const sortedFrames = files
            .filter(file => path.extname(file).toLowerCase() === '.png')
            .sort((a, b) => {
                const aNum = parseInt(path.basename(a).split('_')[1], 10);
                const bNum = parseInt(path.basename(b).split('_')[1], 10);
                return aNum - bNum;
            });

        // Generate URLs for the frames
        const frameUrls = sortedFrames.map(file => `/frames/${path.parse(gifName).name}/${file}`);

        res.json(frameUrls);
    });
});

// API endpoint to list all sb_references JPG Files
app.get('/api/sb_references', (req, res) => {
    const sbRefDir = path.join(__dirname, 'public', 'sb_references');
    fs.readdir(sbRefDir, (err, files) => {
        if (err) {
            console.error('Error reading sb_references directory:', err);
            return res.status(500).json({ error: 'Failed to read sb_references directory.' });
        }
        // Filter JPG files
        const jpgFiles = files.filter(file => path.extname(file).toLowerCase() === '.jpg');
        res.json(jpgFiles);
    });
});

// **New API Endpoint: Get Notes for a Specific Frame of a GIF**
app.get('/api/gifs/:gifName/frames/:frameName/notes', (req, res) => {
    const gifName = req.params.gifName;
    const frameName = req.params.frameName;

    const notes = readNotes();

    if (notes[gifName] && notes[gifName][frameName]) {
        res.json(notes[gifName][frameName]);
    } else {
        res.json([]); // No notes found
    }
});

// **API Endpoint: Add a Note to a Specific Frame of a GIF**
app.post('/api/gifs/:gifName/frames/:frameName/notes', (req, res) => {
    const gifName = req.params.gifName;
    const frameName = req.params.frameName;
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

    // **Create a new note with a unique ID and timestamp**
    const newNote = {
        id: uuidv4(), // Assign a unique ID
        timestamp: new Date().toISOString(),
        content: content.trim()
    };

    notes[gifName][frameName].push(newNote);
    writeNotes(notes);

    console.log(`Added new note to ${gifName} - ${frameName}:`, newNote);
    res.json(newNote);
});

// **API Endpoint: Delete a Specific Note**
app.delete('/api/gifs/:gifName/frames/:frameName/notes/:noteId', (req, res) => {
    const gifName = req.params.gifName;
    const frameName = req.params.frameName;
    const noteId = req.params.noteId;

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
});

// **New API Endpoint: Get All Notes**
app.get('/api/notes', (req, res) => {
    const notes = readNotes();
    res.json(notes);
});

// Start the server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`Server is running at http://localhost:${PORT}`);
});
