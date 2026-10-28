import rawPlugin from 'vite-raw-plugin';
import { defineConfig } from 'vite'

const repositoryBase = process.env.GITHUB_ACTIONS_BASE?.replace(/^\/+|\/+$/g, '');

export default defineConfig({
    build: {
        target: 'esnext'
    },
    base: repositoryBase ? `/${repositoryBase}/` : '/',
    plugins: [
        rawPlugin({
            fileRegex: /\.wgsl$/,
        }),
    ],
})
