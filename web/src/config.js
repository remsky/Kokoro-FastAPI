/**
 * Configuration for API endpoints
 * Fetches root path from server to honor UVICORN_ROOT_PATH
 */

class Config {
    constructor() {
        this.rootPath = '';
        this.initialized = false;
        this.initPromise = this.initialize();
    }
    
    async initialize() {
        try {
            // First detect root path from current URL
            this.detectRootPath();
            
            // Then try to fetch server config to get the actual UVICORN_ROOT_PATH
            // Use the detected root path to build the config URL
            const configUrl = `${this.rootPath}/web/config`;
            const response = await fetch(configUrl);
            if (response.ok) {
                const serverConfig = await response.json();
                // Override with server's root path if provided
                if (serverConfig.root_path !== undefined) {
                    this.rootPath = serverConfig.root_path.replace(/\/$/, '');
                    console.log('Config loaded from server. Root path:', this.rootPath);
                }
            } else {
                console.log('Using detected root path:', this.rootPath);
            }
        } catch (error) {
            console.log('Using detected root path (fetch failed):', this.rootPath, error.message);
        }
        this.initialized = true;
    }
    
    detectRootPath() {
        // Fallback: detect from current URL
        const currentPath = window.location.pathname;
        
        // Extract root path by removing /web/ suffix if present
        let rootPath = '';
        if (currentPath.includes('/web/') || currentPath.endsWith('/web')) {
            const webIndex = currentPath.indexOf('/web');
            rootPath = currentPath.substring(0, webIndex);
        } else if (currentPath.includes('/web')) {
            rootPath = currentPath.split('/web')[0];
        }
        
        this.rootPath = rootPath.replace(/\/$/, '');
        console.log('Config initialized with detected rootPath:', this.rootPath);
    }
    
    async ensureInitialized() {
        if (!this.initialized) {
            await this.initPromise;
        }
    }
    
    /**
     * Get the full API URL for a given endpoint
     * @param {string} endpoint - The endpoint path (e.g., '/v1/audio/speech')
     * @returns {Promise<string>} The full URL with root path
     */
    async getApiUrl(endpoint) {
        await this.ensureInitialized();
        
        // Ensure endpoint starts with /
        if (!endpoint.startsWith('/')) {
            endpoint = '/' + endpoint;
        }
        
        return `${this.rootPath}${endpoint}`;
    }
    
    /**
     * Get the root path
     * @returns {Promise<string>} The root path
     */
    async getRootPath() {
        await this.ensureInitialized();
        return this.rootPath;
    }
}

// Export a singleton instance
export const config = new Config();
export default config;
