const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Serve static files from the current directory
app.use(express.static('.'));

// Handle all routes by serving index.html (for SPA behavior)
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(PORT, () => {
    console.log(`🚀 iQore Frontend Server running on http://localhost:${PORT}`);
    console.log(`📡 Backend API: https://iqoregpt-529970623471.europe-west1.run.app`);
    console.log(`💬 Chat interface ready!`);
}); 