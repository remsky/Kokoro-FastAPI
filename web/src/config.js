// endpoints are built from the server's root path, so a UVICORN_ROOT_PATH mount still resolves

class Config {
    constructor() {
        this.rootPath = '';
        this.version = '';
        this.initialized = false;
        this.initPromise = this.initialize();
    }
    
    async initialize() {
        try {
            this.detectRootPath();

            const configUrl = `${this.rootPath}/web/config`;
            const response = await fetch(configUrl);
            if (response.ok) {
                const serverConfig = await response.json();
                if (serverConfig.root_path !== undefined) {
                    this.rootPath = serverConfig.root_path.replace(/\/$/, '');
                    console.log('Config loaded from server. Root path:', this.rootPath);
                }
                if (serverConfig.version !== undefined) {
                    this.version = serverConfig.version;
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
        const currentPath = window.location.pathname;

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
    
    async getApiUrl(endpoint) {
        await this.ensureInitialized();

        if (!endpoint.startsWith('/')) {
            endpoint = '/' + endpoint;
        }

        return `${this.rootPath}${endpoint}`;
    }
}

export const config = new Config();
export default config;
