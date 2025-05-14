//! NLP Plugin for Luma
//!
//! This plugin provides natural language processing capabilities to Luma,
//! including tokenization, sentence splitting, basic sentiment analysis,
//! and text classification.

use crate::plugins::{register_plugin, PluginResult};
use std::collections::HashMap;

/// Register the NLP plugin with Luma
pub fn register_nlp_plugin() -> Result<(), String> {
    // List of commands supported by this plugin
    let commands = vec![
        "tokenize".to_string(),
        "analyze_sentiment".to_string(),
        "extract_entities".to_string(),
        "summarize".to_string(),
    ];
    
    // Register the plugin
    register_plugin(
        "nlp",
        "Natural Language Processing",
        "1.0.0",
        Some("Provides NLP functionality for text analysis"),
        Some("Luma Team"),
        commands,
        execute_nlp_command,
    )
}

/// Execute an NLP plugin command
fn execute_nlp_command(args: &[&str]) -> PluginResult {
    if args.is_empty() {
        return Err("No NLP command specified".to_string());
    }
    
    match args[0] {
        "tokenize" => tokenize(&args[1..]),
        "analyze_sentiment" => analyze_sentiment(&args[1..]),
        "extract_entities" => extract_entities(&args[1..]),
        "summarize" => summarize(&args[1..]),
        _ => Err(format!("Unknown NLP command: {}", args[0])),
    }
}

/// Split text into tokens
fn tokenize(args: &[&str]) -> PluginResult {
    if args.is_empty() {
        return Err("No text provided for tokenization".to_string());
    }
    
    let text = args.join(" ");
    
    // Simple whitespace and punctuation tokenization
    let mut tokens = Vec::new();
    let mut current_token = String::new();
    
    for ch in text.chars() {
        if ch.is_whitespace() || ch.is_ascii_punctuation() {
            if !current_token.is_empty() {
                tokens.push(current_token.clone());
                current_token.clear();
            }
            
            // Add punctuation as separate tokens
            if ch.is_ascii_punctuation() {
                tokens.push(ch.to_string());
            }
        } else {
            current_token.push(ch);
        }
    }
    
    // Add the last token if any
    if !current_token.is_empty() {
        tokens.push(current_token);
    }
    
    Ok(format!("Tokens: {}", tokens.join(" | ")))
}

/// Perform basic sentiment analysis on text
fn analyze_sentiment(args: &[&str]) -> PluginResult {
    if args.is_empty() {
        return Err("No text provided for sentiment analysis".to_string());
    }
    
    let text = args.join(" ").to_lowercase();
    
    // Simple lexicon-based sentiment analysis
    let positive_words = [
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "happy", "joy", "love", "like", "positive", "best", "better",
    ];
    
    let negative_words = [
        "bad", "terrible", "awful", "horrible", "poor", "negative",
        "sad", "angry", "hate", "dislike", "worst", "worse", "disappointing",
    ];
    
    let words: Vec<&str> = text.split_whitespace().collect();
    
    let mut positive_count = 0;
    let mut negative_count = 0;
    
    for word in words {
        if positive_words.contains(&word) {
            positive_count += 1;
        } else if negative_words.contains(&word) {
            negative_count += 1;
        }
    }
    
    let sentiment = if positive_count > negative_count {
        "positive"
    } else if negative_count > positive_count {
        "negative"
    } else {
        "neutral"
    };
    
    Ok(format!(
        "Sentiment Analysis: {} (positive words: {}, negative words: {})",
        sentiment, positive_count, negative_count
    ))
}

/// Extract named entities from text
fn extract_entities(args: &[&str]) -> PluginResult {
    if args.is_empty() {
        return Err("No text provided for entity extraction".to_string());
    }
    
    let text = args.join(" ");
    
    // Simple rule-based entity extraction
    // Look for capitalized words as potential entities
    let mut entities = Vec::new();
    let words: Vec<&str> = text.split_whitespace().collect();
    
    let mut i = 0;
    while i < words.len() {
        if let Some(first_char) = words[i].chars().next() {
            if first_char.is_uppercase() {
                // Found a potential entity
                let mut entity = words[i].to_string();
                let mut j = i + 1;
                
                // Check if this is part of a multi-word entity
                while j < words.len() {
                    if let Some(next_first_char) = words[j].chars().next() {
                        if next_first_char.is_uppercase() {
                            entity.push(' ');
                            entity.push_str(words[j]);
                            j += 1;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                
                entities.push(entity);
                i = j;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    
    if entities.is_empty() {
        Ok("No entities found in the text".to_string())
    } else {
        Ok(format!("Entities found: {}", entities.join(", ")))
    }
}

/// Produce a simple summary of text
fn summarize(args: &[&str]) -> PluginResult {
    if args.is_empty() {
        return Err("No text provided for summarization".to_string());
    }
    
    let text = args.join(" ");
    
    // Split into sentences
    let sentences: Vec<&str> = text.split(|c| c == '.' || c == '!' || c == '?')
        .filter(|s| !s.trim().is_empty())
        .collect();
    
    if sentences.is_empty() {
        return Err("No valid sentences found in the text".to_string());
    }
    
    // Simple extractive summarization:
    // 1. Count word frequency
    // 2. Score sentences based on word importance
    // 3. Take the top sentences
    
    // Step 1: Count word frequency
    let mut word_freq = HashMap::new();
    for sentence in &sentences {
        for word in sentence.split_whitespace() {
            let word = word.trim().to_lowercase();
            if word.len() > 3 {  // Ignore short words
                *word_freq.entry(word).or_insert(0) += 1;
            }
        }
    }
    
    // Step 2: Score sentences
    let mut sentence_scores = Vec::new();
    for (i, sentence) in sentences.iter().enumerate() {
        let mut score = 0;
        for word in sentence.split_whitespace() {
            let word = word.trim().to_lowercase();
            if let Some(freq) = word_freq.get(&word) {
                score += freq;
            }
        }
        sentence_scores.push((i, score));
    }
    
    // Step 3: Sort and take top sentences
    sentence_scores.sort_by(|a, b| b.1.cmp(&a.1));  // Sort by score descending
    
    // Take 1/3 of the sentences or at least 1
    let summary_count = (sentences.len() / 3).max(1);
    let mut summary_indices: Vec<_> = sentence_scores.iter()
        .take(summary_count)
        .map(|(i, _)| *i)
        .collect();
    
    // Sort by original position
    summary_indices.sort();
    
    // Construct summary
    let summary: Vec<_> = summary_indices.iter()
        .map(|&i| sentences[i].trim())
        .collect();
    
    Ok(format!("Summary: {}", summary.join(". ")))
}