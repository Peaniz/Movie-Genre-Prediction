// filepath: d:\Huukhoa\Nam3\Ki2\DataScience\DeTaiCuoiKi\MiniCrawler\main.js
import os from 'os';
import { crawlMovies } from './worker.js';
import * as genres from './genres/index.js';

// Configuration
const MAX_CONCURRENT_JOBS = Math.max(1, 4); // Leave 1 CPU for OS tasks

/**
 * Main function to crawl all genres
 */
async function main() {
  console.log('Starting MiniCrawler...');
  console.log(`Using up to ${MAX_CONCURRENT_JOBS} concurrent workers`);

  // Get all genre arrays from the imported module
  const genreEntries = Object.entries(genres);
  console.log(`Found ${genreEntries.length} genres to crawl`);

  // Process genres in batches to control concurrency
  const results = [];
  
  for (let i = 0; i < genreEntries.length; i += MAX_CONCURRENT_JOBS) {
    const batch = genreEntries.slice(i, i + MAX_CONCURRENT_JOBS);
    
    console.log(`Processing batch of ${batch.length} genres (${i+1} to ${Math.min(i + batch.length, genreEntries.length)} of ${genreEntries.length})`);
    
    const batchPromises = batch.map(async ([genreName, urls]) => {
      console.log(`Starting crawl for genre: ${genreName} with ${urls.length} URLs`);
      try {
        const result = await crawlMovies(urls, genreName);
        return result;
      } catch (error) {
        console.error(`Error crawling genre ${genreName}:`, error);
        return { genre: genreName, error: error.message };
      }
    });
    
    const batchResults = await Promise.all(batchPromises);
    results.push(...batchResults);
  }
  
  // Print summary
  console.log('\n==== Crawling Summary ====');
  results.forEach(result => {
    if (result.error) {
      console.log(`❌ ${result.genre}: Error - ${result.error}`);
    } else {
      console.log(`✅ ${result.genre}: ${result.count} movies saved to ${result.path}`);
    }
  });
  
  console.log('\nCrawling completed!');
}

// Run the main function
main().catch(error => {
  console.error('Unhandled error in main process:', error);
  process.exit(1);
});