import { defineConfig } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

export default defineConfig(({ mode }) => {
    const singleFileMode = mode === 'singlefile';

    return {
        base: './',
        plugins: singleFileMode ? [viteSingleFile()] : [],
        server: {
            host: true,
            port: 5173,
        },
        build: {
            assetsInlineLimit: singleFileMode ? Number.MAX_SAFE_INTEGER : undefined,
        },
    };
});
