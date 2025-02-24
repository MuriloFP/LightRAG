const CACHE_NAME = 'lightrag-cache';
const API_CACHE_NAME = 'lightrag-api-cache';

// Files to cache for offline access
const CACHE_FILES = [
    '/',
    '/index.html',
    '/static/js/main.js',
    '/static/css/main.css'
];

// Install event - cache static files
self.addEventListener('install', (event) => {
    event.waitUntil(
        Promise.all([
            // Cache static files
            caches.open(CACHE_NAME).then((cache) => {
                return cache.addAll(CACHE_FILES);
            }),
            // Create API cache
            caches.open(API_CACHE_NAME)
        ])
    );
});

// Activate event - cleanup old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (cacheName !== CACHE_NAME && cacheName !== API_CACHE_NAME) {
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
});

// Fetch event - handle offline access
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);
    
    // Handle API requests
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(
            fetch(event.request)
                .then((response) => {
                    // Cache successful responses
                    if (response.ok) {
                        const responseClone = response.clone();
                        caches.open(API_CACHE_NAME).then((cache) => {
                            cache.put(event.request, responseClone);
                        });
                    }
                    return response;
                })
                .catch(() => {
                    // Return cached response if offline
                    return caches.match(event.request);
                })
        );
        return;
    }
    
    // Handle static files
    event.respondWith(
        caches.match(event.request)
            .then((response) => {
                if (response) {
                    return response;
                }
                return fetch(event.request);
            })
    );
});

// Handle messages from the client
self.addEventListener('message', (event) => {
    const operation = event.data;
    
    event.waitUntil(
        caches.open(API_CACHE_NAME).then(async (cache) => {
            switch (operation.type) {
                case 'set':
                    const request = new Request(operation.key);
                    const response = new Response(JSON.stringify(operation.value));
                    await cache.put(request, response);
                    break;
                    
                case 'delete':
                    await cache.delete(new Request(operation.key));
                    break;
                    
                case 'clear':
                    const keys = await cache.keys();
                    await Promise.all(keys.map(key => cache.delete(key)));
                    break;
            }
        })
    );
});

// Handle sync events
self.addEventListener('sync', (event) => {
    if (event.tag === 'sync-cache') {
        event.waitUntil(syncCache());
    }
});

// Sync cache with server
async function syncCache() {
    const cache = await caches.open(API_CACHE_NAME);
    const keys = await cache.keys();
    
    for (const request of keys) {
        try {
            // Attempt to re-fetch and update cache
            const response = await fetch(request);
            if (response.ok) {
                await cache.put(request, response);
            }
        } catch (error) {
            console.error('Failed to sync cache entry:', error);
        }
    }
}

// Handle conflict resolution
async function resolveConflict(key, serverValue, localValue) {
    // Default to server value for now
    // TODO: Implement more sophisticated conflict resolution
    return serverValue;
} 