'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Upload, Search, MessageCircle, Download, BarChart3, FileText } from 'lucide-react';

interface Product {
  id: string;
  product_id: string;
  category: string;
  style_name: string;
  description: string;
  fabric_description: string;
  price: number;
  image_url: string;
  product_link: string;
  created_at: string;
}

interface UploadResult {
  success: boolean;
  message: string;
  upload_id: string;
  storage_method: string;
  row_count: number;
  file_name: string;
  image_urls_found?: number;
}

interface SearchResult {
  success: boolean;
  query: string;
  filters: any;
  results_count: number;
  products: Product[];
}

interface ChatResult {
  success: boolean;
  query: string;
  ai_response?: {
    response: string;
    products_mentioned: string[];
    category: string;
    confidence: number;
  };
  products_considered?: number;
  relevant_products?: any[];
  error?: string;
  suggestion?: string;
}

interface Statistics {
  total_products: number;
  categories: Record<string, number>;
  price_range: {
    min: number;
    max: number;
    avg: number;
  };
  storage_distribution: {
    postgresql: number;
    mongodb: number;
  };
}

const ProductManager: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'upload' | 'search' | 'chat' | 'export' | 'stats'>('upload');
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [searchResults, setSearchResults] = useState<SearchResult | null>(null);
  const [chatResult, setChatResult] = useState<ChatResult | null>(null);
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Form states
  const [searchQuery, setSearchQuery] = useState('');
  const [searchCategory, setSearchCategory] = useState('');
  const [minPrice, setMinPrice] = useState('');
  const [maxPrice, setMaxPrice] = useState('');
  const [chatQuery, setChatQuery] = useState('');
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  useEffect(() => {
    if (activeTab === 'stats') {
      fetchStatistics();
    }
  }, [activeTab]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const endpoint = file.name.endsWith('.csv') ? '/upload/csv' : '/upload/pdf';
      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result: UploadResult = await response.json();
      setUploadResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        query: searchQuery,
        limit: '50',
      });

      if (searchCategory) params.append('category', searchCategory);
      if (minPrice) params.append('min_price', minPrice);
      if (maxPrice) params.append('max_price', maxPrice);

      const response = await fetch(`${API_BASE}/search?${params}`);
      
      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const result: SearchResult = await response.json();
      setSearchResults(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const handleChat = async () => {
    if (!chatQuery.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        query: chatQuery,
        limit: '10',
      });

      const response = await fetch(`${API_BASE}/chat?${params}`);
      
      if (!response.ok) {
        throw new Error(`Chat query failed: ${response.statusText}`);
      }

      const result: ChatResult = await response.json();
      
      // Handle AI disabled case
      if (!result.success && result.error) {
        setError(result.error);
        return;
      }
      
      setChatResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Chat query failed');
    } finally {
      setLoading(false);
    }
  };

  const fetchStatistics = async () => {
    try {
      const response = await fetch(`${API_BASE}/statistics`);
      if (response.ok) {
        const data = await response.json();
        setStatistics(data.statistics);
      }
    } catch (err) {
      console.error('Failed to fetch statistics:', err);
    }
  };

  const handleExport = async (format: 'csv' | 'json') => {
    try {
      const params = new URLSearchParams();
      if (searchCategory) params.append('category', searchCategory);
      if (minPrice) params.append('min_price', minPrice);
      if (maxPrice) params.append('max_price', maxPrice);

      const response = await fetch(`${API_BASE}/export/${format}?${params}`);
      
      if (!response.ok) {
        throw new Error(`Export failed: ${response.statusText}`);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `products_${new Date().toISOString().split('T')[0]}.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed');
    }
  };

  const renderUploadTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4">Upload Product Data</h3>
        
        <div className="space-y-4">
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
            <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <p className="text-gray-600 mb-2">Upload CSV or PDF files with product data</p>
            <p className="text-sm text-gray-500 mb-4">
              Supports: .csv (≤20K rows → PostgreSQL, &gt;20K rows → MongoDB), .pdf (with image URL extraction)
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.pdf"
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? 'Uploading...' : 'Choose File'}
            </button>
          </div>

          {uploadResult && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h4 className="font-semibold text-green-800 mb-2">Upload Successful!</h4>
              <div className="text-sm text-green-700 space-y-1">
                <p><strong>File:</strong> {uploadResult.file_name}</p>
                <p><strong>Rows:</strong> {uploadResult.row_count}</p>
                <p><strong>Storage:</strong> {uploadResult.storage_method}</p>
                <p><strong>Message:</strong> {uploadResult.message}</p>
                {uploadResult.image_urls_found && (
                  <p><strong>Images Found:</strong> {uploadResult.image_urls_found}</p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const renderSearchTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4">Search Products</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <input
            type="text"
            placeholder="Search query..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="border border-gray-300 rounded-md px-3 py-2"
          />
          <select
            value={searchCategory}
            onChange={(e) => setSearchCategory(e.target.value)}
            className="border border-gray-300 rounded-md px-3 py-2"
          >
            <option value="">All Categories</option>
            <option value="Jacket">Jacket</option>
            <option value="Shirt">Shirt</option>
            <option value="T-Shirt">T-Shirt</option>
            <option value="Hoodie">Hoodie</option>
            <option value="Corset">Corset</option>
            <option value="Bottoms">Bottoms</option>
          </select>
          <input
            type="number"
            placeholder="Min price"
            value={minPrice}
            onChange={(e) => setMinPrice(e.target.value)}
            className="border border-gray-300 rounded-md px-3 py-2"
          />
          <input
            type="number"
            placeholder="Max price"
            value={maxPrice}
            onChange={(e) => setMaxPrice(e.target.value)}
            className="border border-gray-300 rounded-md px-3 py-2"
          />
        </div>

        <button
          onClick={handleSearch}
          disabled={loading || !searchQuery.trim()}
          className="bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? 'Searching...' : 'Search'}
        </button>

        {searchResults && (
          <div className="mt-6">
            <h4 className="font-semibold mb-4">
              Results ({searchResults.results_count} products found)
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {searchResults.products.map((product) => (
                <div key={product.id} className="border border-gray-200 rounded-lg p-4">
                  <h5 className="font-semibold text-lg">{product.style_name}</h5>
                  <p className="text-gray-600 text-sm mb-2">{product.category}</p>
                  <p className="text-green-600 font-semibold">₹{product.price}</p>
                  <p className="text-gray-700 text-sm mt-2 line-clamp-3">
                    {product.description}
                  </p>
                  {product.image_url && (
                    <img
                      src={product.image_url}
                      alt={product.style_name}
                      className="w-full h-32 object-cover rounded mt-2"
                    />
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const renderChatTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4">AI Product Assistant</h3>
        
        <div className="space-y-4">
          <div className="flex gap-2">
            <input
              type="text"
              placeholder="Ask about products, styling, recommendations..."
              value={chatQuery}
              onChange={(e) => setChatQuery(e.target.value)}
              className="flex-1 border border-gray-300 rounded-md px-3 py-2"
            />
            <button
              onClick={handleChat}
              disabled={loading || !chatQuery.trim()}
              className="bg-green-600 text-white px-6 py-2 rounded-md hover:bg-green-700 disabled:opacity-50"
            >
              {loading ? 'Thinking...' : 'Ask'}
            </button>
          </div>

          {chatResult && (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <h4 className="font-semibold mb-2">AI Response</h4>
              <div className="prose max-w-none">
                {chatResult.ai_response ? (
                  <>
                    <p className="text-gray-800 mb-4">{chatResult.ai_response.response}</p>
                    <div className="text-sm text-gray-600">
                      <p><strong>Confidence:</strong> {(chatResult.ai_response.confidence * 100).toFixed(1)}%</p>
                      <p><strong>Category:</strong> {chatResult.ai_response.category}</p>
                      <p><strong>Products Considered:</strong> {chatResult.products_considered || 0}</p>
                    </div>
                  </>
                ) : (
                  <div className="text-red-600">
                    <p>{chatResult.error}</p>
                    {chatResult.suggestion && (
                      <p className="text-sm text-gray-600 mt-2">{chatResult.suggestion}</p>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const renderExportTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4">Export Products</h3>
        
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <select
              value={searchCategory}
              onChange={(e) => setSearchCategory(e.target.value)}
              className="border border-gray-300 rounded-md px-3 py-2"
            >
              <option value="">All Categories</option>
              <option value="Jacket">Jacket</option>
              <option value="Shirt">Shirt</option>
              <option value="T-Shirt">T-Shirt</option>
              <option value="Hoodie">Hoodie</option>
              <option value="Corset">Corset</option>
              <option value="Bottoms">Bottoms</option>
            </select>
            <input
              type="number"
              placeholder="Min price"
              value={minPrice}
              onChange={(e) => setMinPrice(e.target.value)}
              className="border border-gray-300 rounded-md px-3 py-2"
            />
            <input
              type="number"
              placeholder="Max price"
              value={maxPrice}
              onChange={(e) => setMaxPrice(e.target.value)}
              className="border border-gray-300 rounded-md px-3 py-2"
            />
          </div>

          <div className="flex gap-4">
            <button
              onClick={() => handleExport('csv')}
              className="bg-green-600 text-white px-6 py-2 rounded-md hover:bg-green-700 flex items-center gap-2"
            >
              <Download className="h-4 w-4" />
              Export CSV
            </button>
            <button
              onClick={() => handleExport('json')}
              className="bg-purple-600 text-white px-6 py-2 rounded-md hover:bg-purple-700 flex items-center gap-2"
            >
              <FileText className="h-4 w-4" />
              Export JSON
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  const renderStatsTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4">Product Statistics</h3>
        
        {statistics ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="text-blue-800 font-semibold">Total Products</h4>
              <p className="text-2xl font-bold text-blue-600">{statistics.total_products}</p>
            </div>
            
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h4 className="text-green-800 font-semibold">PostgreSQL</h4>
              <p className="text-2xl font-bold text-green-600">{statistics.storage_distribution.postgresql}</p>
            </div>
            
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <h4 className="text-purple-800 font-semibold">MongoDB</h4>
              <p className="text-2xl font-bold text-purple-600">{statistics.storage_distribution.mongodb}</p>
            </div>
            
            <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
              <h4 className="text-orange-800 font-semibold">Avg Price</h4>
              <p className="text-2xl font-bold text-orange-600">₹{statistics.price_range.avg.toFixed(0)}</p>
            </div>
          </div>
        ) : (
          <p className="text-gray-600">Loading statistics...</p>
        )}

        {statistics && (
          <div className="mt-6">
            <h4 className="font-semibold mb-4">Category Distribution</h4>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {Object.entries(statistics.categories).map(([category, count]) => (
                <div key={category} className="bg-gray-50 border border-gray-200 rounded-lg p-3">
                  <p className="font-semibold">{category}</p>
                  <p className="text-gray-600">{count} products</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto py-8 px-4">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Product Management System</h1>
          <p className="text-gray-600">
            Upload, search, and manage product data with intelligent storage routing
          </p>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <p className="text-red-800">{error}</p>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="bg-white rounded-lg shadow-md mb-6">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8 px-6">
              {[
                { id: 'upload', label: 'Upload', icon: Upload },
                { id: 'search', label: 'Search', icon: Search },
                { id: 'chat', label: 'AI Chat', icon: MessageCircle },
                { id: 'export', label: 'Export', icon: Download },
                { id: 'stats', label: 'Statistics', icon: BarChart3 },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <tab.icon className="h-4 w-4" />
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'upload' && renderUploadTab()}
        {activeTab === 'search' && renderSearchTab()}
        {activeTab === 'chat' && renderChatTab()}
        {activeTab === 'export' && renderExportTab()}
        {activeTab === 'stats' && renderStatsTab()}
      </div>
    </div>
  );
};

export default ProductManager; 