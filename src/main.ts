import './styles.css';
import { initApp } from './app';

const mountEl = document.querySelector<HTMLDivElement>('#app');

if (!mountEl) {
    throw new Error('Missing #app mount element');
}

initApp(mountEl);
