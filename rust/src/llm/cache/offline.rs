use std::sync::Arc;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{ServiceWorkerRegistration, ServiceWorkerContainer, ServiceWorker};
use super::backend::CacheError;
use super::types::{CacheEntry, CacheValue};
use serde::{Serialize, Deserialize};
use std::collections::VecDeque;
use parking_lot::Mutex;

const SW_SCRIPT: &str = "/sw.js";  // Path to service worker script

/// Offline operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OfflineOperation {
    Set(CacheEntry),
    Delete(String),
    Clear,
}

/// Pending operation for offline sync
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingOperation {
    /// Operation type
    pub operation: OfflineOperation,
    /// Timestamp when operation was created
    pub timestamp: f64,
    /// Version at time of operation
    pub version: u64,
}

/// Offline sync queue
#[derive(Debug)]
pub struct SyncQueue {
    /// Queue of pending operations
    operations: VecDeque<PendingOperation>,
    /// Current version
    version: u64,
}

impl SyncQueue {
    pub fn new() -> Self {
        Self {
            operations: VecDeque::new(),
            version: 0,
        }
    }

    pub fn push(&mut self, operation: OfflineOperation) {
        let pending = PendingOperation {
            operation,
            timestamp: js_sys::Date::now(),
            version: self.version,
        };
        self.operations.push_back(pending);
    }

    pub fn pop(&mut self) -> Option<PendingOperation> {
        self.operations.pop_front()
    }

    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    pub fn increment_version(&mut self) {
        self.version += 1;
    }
}

/// Offline support manager
pub struct OfflineManager {
    /// Service worker registration
    registration: Option<ServiceWorkerRegistration>,
    /// Sync queue
    sync_queue: Arc<Mutex<SyncQueue>>,
    /// Online status
    is_online: bool,
}

impl OfflineManager {
    /// Create a new offline manager
    pub async fn new() -> Result<Self, CacheError> {
        let window = web_sys::window()
            .ok_or_else(|| CacheError::InitError("No window object available".into()))?;
            
        let navigator = window.navigator();
        let sw = navigator.service_worker();
        
        let registration = Self::register_service_worker(&sw).await?;
        
        Ok(Self {
            registration: Some(registration),
            sync_queue: Arc::new(Mutex::new(SyncQueue::new())),
            is_online: navigator.online(),
        })
    }
    
    /// Register the service worker
    async fn register_service_worker(container: &ServiceWorkerContainer) -> Result<ServiceWorkerRegistration, CacheError> {
        let promise = container.register(SW_SCRIPT)
            .map_err(|e| CacheError::InitError(format!("Failed to register service worker: {:?}", e)))?;
            
        let registration = JsFuture::from(promise)
            .await
            .map_err(|e| CacheError::InitError(format!("Service worker registration failed: {:?}", e)))?;
            
        let registration: ServiceWorkerRegistration = registration
            .dyn_into()
            .map_err(|_| CacheError::InitError("Invalid registration object".into()))?;
            
        Ok(registration)
    }
    
    /// Queue an operation for offline sync
    pub fn queue_operation(&self, operation: OfflineOperation) {
        let mut queue = self.sync_queue.lock();
        queue.push(operation);
    }
    
    /// Process queued operations
    pub async fn process_queue(&mut self) -> Result<(), CacheError> {
        if !self.is_online {
            return Ok(());
        }
        
        let mut queue = self.sync_queue.lock();
        while let Some(operation) = queue.pop() {
            self.process_operation(operation).await?;
        }
        
        queue.increment_version();
        Ok(())
    }
    
    /// Process a single operation
    async fn process_operation(&self, operation: PendingOperation) -> Result<(), CacheError> {
        // Send operation to service worker
        if let Some(registration) = &self.registration {
            let active = registration.active()
                .ok_or_else(|| CacheError::Unavailable("No active service worker".into()))?;
                
            let msg = serde_wasm_bindgen::to_value(&operation)
                .map_err(|e| CacheError::StorageError(format!("Failed to serialize operation: {:?}", e)))?;
                
            active.post_message(&msg)
                .map_err(|e| CacheError::StorageError(format!("Failed to send message to service worker: {:?}", e)))?;
        }
        
        Ok(())
    }
    
    /// Update online status
    pub fn set_online_status(&mut self, online: bool) {
        self.is_online = online;
        if online {
            // Schedule queue processing when we come online
            let queue = Arc::clone(&self.sync_queue);
            wasm_bindgen_futures::spawn_local(async move {
                let mut manager = Self {
                    registration: None,
                    sync_queue: queue,
                    is_online: true,
                };
                if let Err(e) = manager.process_queue().await {
                    web_sys::console::error_1(&format!("Failed to process queue: {:?}", e).into());
                }
            });
        }
    }
    
    /// Check if we're online
    pub fn is_online(&self) -> bool {
        self.is_online
    }
}

/// Service worker message handler
#[wasm_bindgen]
pub async fn handle_sw_message(event: web_sys::MessageEvent) -> Result<(), JsValue> {
    let data = event.data();
    let operation: PendingOperation = serde_wasm_bindgen::from_value(data)?;
    
    match operation.operation {
        OfflineOperation::Set(entry) => {
            // Handle cache set
            let cache = web_sys::window()
                .unwrap()
                .caches()
                .unwrap()
                .open("lightrag-cache")
                .await?;
                
            let request = web_sys::Request::new_with_str(&entry.key)?;
            let response = web_sys::Response::new_with_str(&serde_json::to_string(&entry).unwrap())?;
            cache.put_with_request_and_response(&request, &response).await?;
        }
        OfflineOperation::Delete(key) => {
            // Handle cache delete
            let cache = web_sys::window()
                .unwrap()
                .caches()
                .unwrap()
                .open("lightrag-cache")
                .await?;
                
            let request = web_sys::Request::new_with_str(&key)?;
            cache.delete_with_request(&request).await?;
        }
        OfflineOperation::Clear => {
            // Handle cache clear
            let cache = web_sys::window()
                .unwrap()
                .caches()
                .unwrap()
                .open("lightrag-cache")
                .await?;
                
            let keys = cache.keys().await?;
            for i in 0..keys.length() {
                if let Some(request) = keys.get(i) {
                    cache.delete_with_request(&request).await?;
                }
            }
        }
    }
    
    Ok(())
} 