// server.js

const express = require('express');
const path = require('path');
const multer = require('multer');
const fs = require('fs');
const bodyParser = require('body-parser');
const app = express();
const { v4: uuidv4 } = require('uuid');


// Suppose you store them in selected_frames.json
const selectedFramesFile = path.join(__dirname, 'selected_frames.json');

// Path to the notes.json file
const notesFilePath = path.join(__dirname, 'notes.json');

// A cache to store all MoCap frames by GIF base name
const mocapFrameCache = {};


// Middleware to parse JSON bodies
app.use(bodyParser.json());


// Serve static files from the "public" directory
app.use(express.static(path.join(__dirname, 'public')));


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

// API endpoint to list all 3D MoCap GIFs
app.get('/api/mocap_gifs', (req, res) => {
    const mocapGifsDir = path.join(__dirname, 'public', 'mocap_gifs');
    fs.readdir(mocapGifsDir, (err, files) => {
        if (err) {
            console.error('Error reading MoCap GIFs directory:', err);
            return res.status(500).json({ error: 'Unable to scan MoCap GIFs directory.' });
        }
        // Filter only GIF files
        const gifFiles = files.filter(file => path.extname(file).toLowerCase() === '.gif');
        res.json(gifFiles);
    });
});

// API endpoint to get frames for a specific 3D MoCap GIF
app.get('/api/mocap_gifs/:gifName/frames', (req, res) => {
    const gifName = req.params.gifName;
    // We remove the .gif extension if you’re storing frames in a folder named just by the base name
    const baseName = path.parse(gifName).name;

    // Construct the folder path for the frames
    const mocapFramesDir = path.join(__dirname, 'public', 'mocap_frames', baseName);

    // Check if frames directory exists
    if (!fs.existsSync(mocapFramesDir)) {
        console.warn(`MoCap frames directory not found for GIF: ${gifName}`);
        return res.status(404).json({ error: 'Frames not found for the selected MoCap GIF.' });
    }

    // Read frame files in that directory
    fs.readdir(mocapFramesDir, (err, files) => {
        if (err) {
            console.error('Error reading MoCap frames directory:', err);
            return res.status(500).json({ error: 'Unable to read MoCap frames directory.' });
        }

        // Filter out only .png files (assuming frames are PNG)
        const sortedFrames = files
            .filter(file => path.extname(file).toLowerCase() === '.png')
            // Sort numerically if your file naming is like frame_1.png, frame_2.png, etc.
            .sort((a, b) => {
                const aNum = parseInt(path.basename(a).split('_')[1], 10);
                const bNum = parseInt(path.basename(b).split('_')[1], 10);
                return aNum - bNum;
            });

        // Construct URLs for each frame. 
        // Because of app.use(express.static('public')), these files are available at /mocap_frames/<gifName>/<filename>.
        const frameUrls = sortedFrames.map(file => `/mocap_frames/${baseName}/${file}`);

        // Return the list of frame URLs
        res.json(frameUrls);
    });
});

// API endpoint to list all selected frames for all MoCap GIFs
app.get('/api/mocap_gifs/:gifName/selected_frames', (req, res) => {
    const gifName = req.params.gifName; // e.g. "myMoCap.gif"
    const baseName = path.parse(gifName).name; // e.g. "myMoCap"
  
    const allSelected = readSelectedFrames();
    const framesForGif = allSelected[baseName] || [];
    res.json(framesForGif);
  });
  
// API endpoint to add selected frames for a specific MoCap GIF
app.post('/api/mocap_gifs/:gifName/selected_frames', (req, res) => {
    const gifName = req.params.gifName;
    const baseName = path.parse(gifName).name;
  
    // The request body might look like: { rangeOrIndex: "211-234" } or { rangeOrIndex: "210" }
    const { rangeOrIndex } = req.body;
    if (!rangeOrIndex || typeof rangeOrIndex !== 'string') {
      return res.status(400).json({ error: 'Invalid or missing rangeOrIndex.' });
    }
  
    // Convert the rangeOrIndex into an array of integers
    let indexesToAdd = [];
    
    // Check if it's like "211-234"
    if (rangeOrIndex.includes('-')) {
      const [start, end] = rangeOrIndex.split('-').map(str => parseInt(str.trim(), 10));
      if (isNaN(start) || isNaN(end) || start > end) {
        return res.status(400).json({ error: 'Invalid range format.' });
      }
      for (let i = start; i <= end; i++) {
        indexesToAdd.push(i);
      }
    } else {
      // Single index
      const single = parseInt(rangeOrIndex, 10);
      if (isNaN(single)) {
        return res.status(400).json({ error: 'Invalid index format.' });
      }
      indexesToAdd.push(single);
    }
  
    const allSelected = readSelectedFrames();
    // Ensure we have an array for the baseName
    if (!allSelected[baseName]) {
      allSelected[baseName] = [];
    }
  
    // Merge new indexes
    const existing = new Set(allSelected[baseName]);
    indexesToAdd.forEach(idx => existing.add(idx));
    allSelected[baseName] = Array.from(existing).sort((a, b) => a - b); // Keep them sorted
  
    writeSelectedFrames(allSelected);
  
    res.json({ message: 'Frames added successfully.', frames: allSelected[baseName] });
  });

// API endpoint to delete a selected frame for a specific MoCap GIF
// Example: DELETE /api/mocap_gifs/someMoCap.gif/selected_frames?start=211&end=234
app.delete('/api/mocap_gifs/:gifName/selected_frames', (req, res) => {
    const gifName = req.params.gifName;
    const baseName = path.parse(gifName).name;
    
    // Parse query params for range
    let { start, end } = req.query;
    start = parseInt(start, 10);
    end   = parseInt(end,   10);
  
    if (isNaN(start) || isNaN(end) || start > end) {
      return res.status(400).json({ error: 'Invalid start/end range for deletion.' });
    }
  
    const allSelected = readSelectedFrames(); // from selected_frames.json
    if (!allSelected[baseName]) {
      // No frames stored at all
      return res.status(404).json({ error: 'No frames stored for this MoCap GIF.' });
    }
    
    // Filter out indexes in the specified range
    const oldArr = allSelected[baseName];
    const newArr = oldArr.filter(idx => idx < start || idx > end);
    allSelected[baseName] = newArr;
  
    writeSelectedFrames(allSelected);
  
    return res.json({
      message: `Removed frames in range ${start}-${end}`,
      frames: newArr
    });
  });

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

// Utility function to read selected frames from the JSON file
function readSelectedFrames() {
  try {
    if (!fs.existsSync(selectedFramesFile)) {
      fs.writeFileSync(selectedFramesFile, JSON.stringify({}, null, 2), 'utf8');
      return {};
    }
    const data = fs.readFileSync(selectedFramesFile, 'utf8');
    return JSON.parse(data);
  } catch (err) {
    console.error('Error reading selected_frames.json:', err);
    return {};
  }
}

// Utility function to write selected frames to the JSON file
function writeSelectedFrames(data) {
  try {
    fs.writeFileSync(selectedFramesFile, JSON.stringify(data, null, 2), 'utf8');
    console.log('Successfully wrote to selected_frames.json');
  } catch (err) {
    console.error('Error writing to selected_frames.json:', err);
  }
}

app.get('/api/mocap_gifs/:gifName/selected_frames', (req, res) => {
  const gifName = req.params.gifName; // e.g. "myMoCap.gif"
  const baseName = path.parse(gifName).name; // e.g. "myMoCap"

  const allSelected = readSelectedFrames();
  const framesForGif = allSelected[baseName] || [];
  res.json(framesForGif);
});



// HANDELING FILE UPLOAD
const uploadDir = path.join(__dirname, 'public', 'data/mocap');

if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}



const upload = multer({
    dest: path.join(__dirname, 'uploads'), // Temporary upload directory
    limits: { fileSize: 50 * 1024 * 1024 }, // 50 MB limit
});

// Endpoint to upload multiple files
app.post('/api/data/mocap', (req, res, next) => {
    upload.array('mocapFile', 10)(req, res, (err) => { // Allow up to 10 files
        if (err) {
            console.error('Error during file upload:', err);
            return res.status(500).json({ error: 'File upload failed.' });
        }

        const mocapDir = path.join(__dirname, 'public', 'data', 'mocap');
        const uploadedFiles = [];
        const ignoredFiles = [];

        req.files.forEach(file => {
            const targetPath = path.join(mocapDir, file.originalname);

            // Check if the file already exists
            if (fs.existsSync(targetPath)) {
                console.log(`File already exists, ignoring: ${file.originalname}`);
                ignoredFiles.push(file.originalname);

                // Remove temporary uploaded file
                fs.unlinkSync(file.path);
            } else {
                try {
                    // Move the file to the target directory
                    fs.renameSync(file.path, targetPath);
                    console.log(`File uploaded successfully: ${file.originalname}`);
                    uploadedFiles.push({ filename: file.originalname });
                } catch (err) {
                    console.error(`Error moving file ${file.originalname}:`, err);
                }
            }
        });

        res.json({
            message: 'Upload completed.',
            uploadedFiles,
            ignoredFiles,
        });
    });
});


// Endpoint to fetch file list
app.get('/api/data/mocap', (req, res) => {
    const mocapDir = path.join(__dirname, 'public', 'data', 'mocap');

    fs.readdir(mocapDir, (err, files) => {
        if (err) {
            console.error('Error reading MoCap directory:', err);
            return res.status(500).json({ error: 'Failed to fetch file list.' });
        }

        // Filter files based on extensions if needed
        const validExtensions = ['.csv'];
        const filteredFiles = files.filter(file =>
            validExtensions.includes(path.extname(file))
        );

        res.json(filteredFiles);
    });
});

app.delete('/api/data/mocap', (req, res) => {
    const mocapDir = path.join(__dirname, 'public', 'data', 'mocap');
    const { files } = req.body;

    if (!files || files.length === 0) {
        return res.status(400).json({ error: 'No files specified for deletion.' });
    }

    const deletedFiles = [];
    const failedFiles = [];

    files.forEach(file => {
        const filePath = path.join(mocapDir, file);

        if (fs.existsSync(filePath)) {
            try {
                fs.unlinkSync(filePath);
                deletedFiles.push(file);
            } catch (err) {
                console.error(`Error deleting file ${file}:`, err);
                failedFiles.push(file);
            }
        } else {
            failedFiles.push(file);
        }
    });

    res.json({
        deletedFiles,
        failedFiles,
        message: `${deletedFiles.length} files deleted, ${failedFiles.length} failed.`,
    });
});

// Start the server
const PORT = process.env.PORT || 2001;
app.listen(PORT, () => {
    console.log(`Server is running at http://localhost:${PORT}`);
});