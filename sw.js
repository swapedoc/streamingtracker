// StreamIQ Service Worker
// Minimal SW — just enough to make the app installable as a PWA
// No aggressive caching — Supabase data must always be fresh

const CACHE = 'streamiq-v1';
const PRECACHE = ['/'];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(PRECACHE))
  );
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Network-first strategy — always fetch fresh, fall back to cache if offline
self.addEventListener('fetch', e => {
  // Only handle same-origin HTML requests (the app shell)
  if (e.request.mode === 'navigate') {
    e.respondWith(
      fetch(e.request).catch(() => caches.match('/'))
    );
  }
});
