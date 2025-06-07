self.addEventListener('install', function(event) {
  console.log('[Service Worker] Installing Service Worker ...', event);
  // You can add caching logic here for offline access
});

self.addEventListener('activate', function(event) {
  console.log('[Service Worker] Activating Service Worker ...', event);
});

self.addEventListener('fetch', function(event) {
  // You can add more sophisticated caching strategies here
  return fetch(event.request);
});