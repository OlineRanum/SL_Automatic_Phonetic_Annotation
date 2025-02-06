// controllers/mocapGifsController.js
const fs = require('fs');
const path = require('path');
const { readJSON, writeJSON } = require('../utils/fileUtils');

const mocapGifsDir = path.join(__dirname, '..','..', 'public','graphics', 'mocap_gifs');
const mocapFramesDir = path.join(__dirname, '..','..', 'public','graphics', 'mocap_frames');
const selectedFramesFile = path.join(__dirname, '..','..','public','output', 'selected_frames.json');

exports.listMocapGifs = (req, res) => {
    fs.readdir(mocapGifsDir, (err, files) => {
        if (err) {
            console.error('Error reading MoCap GIFs directory:', err);
            return res.status(500).json({ error: 'Unable to scan MoCap GIFs directory.' });
        }
        const gifFiles = files.filter(file => path.extname(file).toLowerCase() === '.gif');
        res.json(gifFiles);
    });
};

exports.getMocapGifFrames = (req, res) => {
    const { gifName } = req.params;
    const baseName = path.parse(gifName).name;
    const framesDir = path.join(mocapFramesDir, baseName);

    if (!fs.existsSync(framesDir)) {
        console.warn(`MoCap frames directory not found for GIF: ${gifName}`);
        return res.status(404).json({ error: 'Frames not found for the selected MoCap GIF.' });
    }

    fs.readdir(framesDir, (err, files) => {
        if (err) {
            console.error('Error reading MoCap frames directory:', err);
            return res.status(500).json({ error: 'Unable to read MoCap frames directory.' });
        }

        const sortedFrames = files
            .filter(file => path.extname(file).toLowerCase() === '.png')
            .sort((a, b) => {
                const aNum = parseInt(path.basename(a).split('_')[1], 10);
                const bNum = parseInt(path.basename(b).split('_')[1], 10);
                return aNum - bNum;
            });

        const frameUrls = sortedFrames.map(file => `/graphics/mocap_frames/${baseName}/${file}`);
        res.json(frameUrls);
    });
};

exports.getSelectedFrames = (req, res) => {
    const { gifName } = req.params;
    const baseName = path.parse(gifName).name;

    const allSelected = readJSON(selectedFramesFile, {});
    const framesForGif = allSelected[baseName] || [];
    res.json(framesForGif);
};

exports.addSelectedFrames = (req, res) => {
    const { gifName } = req.params;
    const baseName = path.parse(gifName).name;
    const { rangeOrIndex } = req.body;

    if (!rangeOrIndex || typeof rangeOrIndex !== 'string') {
        return res.status(400).json({ error: 'Invalid or missing rangeOrIndex.' });
    }

    let indexesToAdd = [];

    if (rangeOrIndex.includes('-')) {
        const [start, end] = rangeOrIndex.split('-').map(str => parseInt(str.trim(), 10));
        if (isNaN(start) || isNaN(end) || start > end) {
            return res.status(400).json({ error: 'Invalid range format.' });
        }
        for (let i = start; i <= end; i++) {
            indexesToAdd.push(i);
        }
    } else {
        const single = parseInt(rangeOrIndex, 10);
        if (isNaN(single)) {
            return res.status(400).json({ error: 'Invalid index format.' });
        }
        indexesToAdd.push(single);
    }

    const allSelected = readJSON(selectedFramesFile, {});
    if (!allSelected[baseName]) {
        allSelected[baseName] = [];
    }

    const existing = new Set(allSelected[baseName]);
    indexesToAdd.forEach(idx => existing.add(idx));
    allSelected[baseName] = Array.from(existing).sort((a, b) => a - b);

    writeJSON(selectedFramesFile, allSelected);

    res.json({ message: 'Frames added successfully.', frames: allSelected[baseName] });
};

exports.deleteSelectedFrames = (req, res) => {
    const { gifName } = req.params;
    const baseName = path.parse(gifName).name;
    let { start, end } = req.query;

    start = parseInt(start, 10);
    end = parseInt(end, 10);

    if (isNaN(start) || isNaN(end) || start > end) {
        return res.status(400).json({ error: 'Invalid start/end range for deletion.' });
    }

    const allSelected = readJSON(selectedFramesFile, {});
    if (!allSelected[baseName]) {
        return res.status(404).json({ error: 'No frames stored for this MoCap GIF.' });
    }

    const newArr = allSelected[baseName].filter(idx => idx < start || idx > end);
    allSelected[baseName] = newArr;

    writeJSON(selectedFramesFile, allSelected);

    res.json({
        message: `Removed frames in range ${start}-${end}`,
        frames: newArr,
    });
};
