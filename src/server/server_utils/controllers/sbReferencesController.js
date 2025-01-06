// controllers/sbReferencesController.js
const fs = require('fs');
const path = require('path');

const sbRefDir = path.join(__dirname, '..','..', 'public', 'sb_references');

exports.listSbReferences = (req, res) => {
    fs.readdir(sbRefDir, (err, files) => {
        if (err) {
            console.error('Error reading sb_references directory:', err);
            return res.status(500).json({ error: 'Failed to read sb_references directory.' });
        }
        const jpgFiles = files.filter(file => path.extname(file).toLowerCase() === '.jpg');
        res.json(jpgFiles);
    });
};
