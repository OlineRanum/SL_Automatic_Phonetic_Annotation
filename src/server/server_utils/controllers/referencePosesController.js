// controllers/referencePosesController.js
const fs = require('fs');
const path = require('path');

const refPosesDir = path.join(__dirname, '..','..', 'public','graphics', 'reference_poses');

exports.listReferencePoses = (req, res) => {
    fs.readdir(refPosesDir, (err, files) => {
        if (err) {
            console.error('Error reading Reference Poses directory:', err);
            return res.status(500).json({ error: 'Unable to scan Reference Poses directory.' });
        }
        const pngFiles = files.filter(file => path.extname(file).toLowerCase() === '.png');
        res.json(pngFiles);
    });
};
