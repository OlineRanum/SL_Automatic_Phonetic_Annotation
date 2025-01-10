// controllers/gifsController.js
const fs = require('fs');
const path = require('path');

const gifsDir = path.join(__dirname, '..',  '..','public','graphics', 'gifs');

exports.listGifs = (req, res) => {
    fs.readdir(gifsDir, (err, files) => {
        if (err) {
            console.error('Error reading GIFs directory:', err);
            return res.status(500).json({ error: 'Unable to scan GIFs directory.' });
        }
        const gifFiles = files.filter(file => path.extname(file).toLowerCase() === '.gif');
        res.json(gifFiles);
    });
};

exports.getGifFrames = (req, res) => {
    const gifName = req.params.gifName;
    const framesDir = path.join(__dirname, '..', '..', 'public', 'graphics','frames', path.parse(gifName).name);

    if (!fs.existsSync(framesDir)) {
        console.warn(`Frames directory not found for GIF: ${gifName}`);
        return res.status(404).json({ error: 'Frames not found for the selected GIF.' });
    }

    fs.readdir(framesDir, (err, files) => {
        if (err) {
            console.error('Error reading frames directory:', err);
            return res.status(500).json({ error: 'Unable to read frames directory.' });
        }

        const sortedFrames = files
            .filter(file => path.extname(file).toLowerCase() === '.png')
            .sort((a, b) => {
                const aNum = parseInt(path.basename(a).split('_')[1], 10);
                const bNum = parseInt(path.basename(b).split('_')[1], 10);
                return aNum - bNum;
            });

        const frameUrls = sortedFrames.map(file => `graphics/frames/${path.parse(gifName).name}/${file}`);
        res.json(frameUrls);
    });
};
