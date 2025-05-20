import puppeteer from 'puppeteer';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Create output directory if it doesn't exist
const outputDir = path.join(__dirname, 'output');
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir);
}

const outputFile = path.join(outputDir, `output-${new Date().toISOString().replaceAll('.', '-').replaceAll(':', '-')}.csv`);

// Initialize the CSV file with headers if it doesn't exist
if (!fs.existsSync(outputFile)) {
    fs.writeFileSync(outputFile, 'id,title,description,genres,genre_category,rating,url\n');
}

/**
 * Worker to crawl movie details from Rotten Tomatoes
 */
export async function crawlMovies(urls, genreName) {
    // Initialize browser with GPU disabled
    const browser = await puppeteer.launch({
        headless: 'new',
        args: [
            '--disable-gpu',
            '--disable-dev-shm-usage',
            '--disable-setuid-sandbox',
            '--no-sandbox',
        ]
    });

    console.log(`Starting to crawl ${urls.length} URLs for genre: ${genreName}`);
    
    const results = [];
    let processedCount = 0;
    
    try {
        // Process URLs in batches to avoid memory issues
        const batchSize = 5;
        for (let i = 0; i < urls.length; i += batchSize) {
            const batch = urls.slice(i, i + batchSize);
            
            const batchPromises = batch.map(async (url) => {
                const page = await browser.newPage();
                // Set a timeout for navigation
                page.setDefaultNavigationTimeout(30000);
                
                try {
                    await page.goto(url, { waitUntil: 'domcontentloaded' });
                    
                    // Extract slug from URL for record ID
                    const slug = url.split('/').pop();
                    
                    // Use the selectors from note.md to extract data
                    const movieData = await page.evaluate(() => {
                        // Initialize with empty values in case some selectors don't match
                        const data = {
                            title: '',
                            description: '',
                            genres: [],
                            rating: ''
                        };
                        
                        try {
                            // Extract title
                            const titleElement = document.querySelector('rt-text[slot="title"]');
                            if (titleElement) {
                                data.title = titleElement.textContent.trim();
                            }
                            
                            // Extract description
                            const descriptionElement = document.querySelector('rt-text[data-qa="synopsis-value"]');
                            if (descriptionElement) {
                                data.description = descriptionElement.textContent.trim();
                            }
                            
                            // Extract genres
                            const genreElements = document.querySelectorAll('rt-text[slot="metadataGenre"]');
                            if (genreElements && genreElements.length > 0) {
                                data.genres = Array.from(genreElements).map(node => 
                                    node.textContent.replace('\/', '').trim()
                                );
                            }
                            
                            // Extract rating
                            const ratingElement = document.querySelector('rt-text[slot="metadataProp"]');
                            if (ratingElement) {
                                const ratingText = ratingElement.textContent.replace(',', '').trim();
                                // Use regex to validate common film ratings (G, PG, PG-13, R, NC-17)
                                const ratingRegex = /^(G|PG|PG-13|R|NC-17)$/;
                                data.rating = ratingRegex.test(ratingText) ? ratingText : "Unknown";
                            } else {
                                data.rating = "Unknown";
                            }
                            
                        } catch (error) {
                            console.error('Error in page.evaluate:', error);
                        }
                        
                        return data;
                    });
                    
                    // Add slug as ID, URL, and genre category to the data
                    movieData.id = slug;
                    movieData.url = url;
                    movieData.genre_category = genreName;
                    
                    results.push(movieData);
                    processedCount++;
                    
                    if (processedCount % 10 === 0 || processedCount === urls.length) {
                        console.log(`Processed ${processedCount}/${urls.length} for genre: ${genreName}`);
                    }
                    
                } catch (error) {
                    console.error(`Error processing ${url}:`, error.message);
                } finally {
                    await page.close();
                }
            });
            
            await Promise.all(batchPromises);
        }
        
        // Append results to the single CSV file
        const csvContent = generateCSVRows(results, genreName);
        fs.appendFileSync(outputFile, csvContent);
        
        console.log(`Completed crawling for genre: ${genreName}. Results appended to ${outputFile}`);
        return { genre: genreName, count: results.length, path: outputFile };
        
    } catch (error) {
        console.error(`Error in crawlMovies for genre ${genreName}:`, error);
        throw error;
    } finally {
        await browser.close();
    }
}

/**
 * Generate CSV rows from movie data (without headers)
 */
function generateCSVRows(movieData, genreName) {
    if (!movieData || movieData.length === 0) {
        return '';
    }
    
    let csvContent = '';
    
    // Add each movie as a row
    movieData.forEach(movie => {
        // Escape fields for CSV
        const escapedTitle = escapeCSVField(movie.title || '');
        const escapedDescription = escapeCSVField(movie.description || '');
        const escapedGenres = escapeCSVField((movie.genres || []).join(', '));
        const escapedRating = escapeCSVField(movie.rating || '');
        const escapedUrl = escapeCSVField(movie.url || '');
        
        csvContent += `${movie.id},${escapedTitle},${escapedDescription},${escapedGenres},${genreName},${escapedRating},${escapedUrl}\n`;
    });
    
    return csvContent;
}

/**
 * Escape a field for CSV
 */
function escapeCSVField(field) {
    if (typeof field !== 'string') {
        field = String(field);
    }
    
    // If the field contains quotes, commas, or newlines, enclose it in quotes and escape inner quotes
    if (field.includes('"') || field.includes(',') || field.includes('\n')) {
        return `"${field.replace(/"/g, '""')}"`;
    }
    
    return field;
}