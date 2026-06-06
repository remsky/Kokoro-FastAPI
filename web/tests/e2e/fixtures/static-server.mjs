import { createReadStream, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize, resolve } from 'node:path';

const port = Number(process.env.PLAYWRIGHT_STATIC_PORT || 4173);
const root = resolve('web');

const contentTypes = {
    '.css': 'text/css',
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.svg': 'image/svg+xml',
};

function resolveRequestPath(url) {
    const pathname = new URL(url, `http://127.0.0.1:${port}`).pathname;
    const relativePath = pathname === '/' ? 'index.html' : pathname.slice(1);
    const requested = resolve(root, normalize(relativePath));

    if (!requested.startsWith(root)) {
        return null;
    }

    return requested;
}

const server = createServer((request, response) => {
    const filePath = resolveRequestPath(request.url);
    if (!filePath) {
        response.writeHead(403);
        response.end();
        return;
    }

    try {
        const stat = statSync(filePath);
        if (!stat.isFile()) {
            throw new Error('Not a file');
        }

        response.writeHead(200, {
            'Content-Length': stat.size,
            'Content-Type': contentTypes[extname(filePath)] || 'application/octet-stream',
        });
        createReadStream(filePath).pipe(response);
    } catch {
        response.writeHead(404);
        response.end();
    }
});

server.listen(port, '127.0.0.1');
