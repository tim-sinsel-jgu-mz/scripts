compare_names_fixed_cols <- function(txt_folder_path, csv_folder_path, output_csv_path) {
  
  # --- Fixed Column Indices ---
  # For TXT files: Surname is 3rd column, Name is 5th column
  txt_surname_col_idx <- 3
  txt_name_col_idx <- 5
  
  # For CSV file: Surname is 1st column, Name is 2nd column
  csv_surname_col_idx <- 1
  csv_name_col_idx <- 2
  # --- End Fixed Column Indices ---
  
  
  # --- 1. Read and collect names from TXT files into a list of vectors ---
  all_txt_names_pairs <- list()
  
  if (!dir.exists(txt_folder_path)) {
    warning("TXT folder does not exist: ", txt_folder_path)
    return(list())
  }
  
  txt_files <- list.files(txt_folder_path, pattern = "\\.txt$", full.names = TRUE, ignore.case = TRUE)
  if (length(txt_files) == 0) {
    warning("No .txt files found in: ", txt_folder_path)
    return(list())
  }
  
  for (file_path in txt_files) {
    tryCatch({
      # Read TXT file: skip 1 line, use 2nd line for headers, tab-separated
      txt_data <- read.delim(file_path, header = TRUE, skip = 1, sep = "\t",
                             stringsAsFactors = FALSE, quote = "", encoding = "latin1")
      
      # Check if column indices are valid for the current TXT file
      num_cols_txt <- ncol(txt_data)
      if (txt_name_col_idx > num_cols_txt || txt_name_col_idx < 1 ||
          txt_surname_col_idx > num_cols_txt || txt_surname_col_idx < 1) {
        warning("Invalid column indices (Name/Surname) for TXT file '", basename(file_path), "'. File has ", num_cols_txt, " columns. Skipping.")
        next # Skip to the next file
      }
      
      names_extracted <- txt_data[[txt_name_col_idx]]
      surnames_extracted <- txt_data[[txt_surname_col_idx]]
      
      # Trim whitespace and double quotes from values
      names_extracted <- gsub('^"|"$', '', trimws(names_extracted))
      surnames_extracted <- gsub('^"|"$', '', trimws(surnames_extracted))
      
      # Remove any rows where Name or Surname became empty after trimming
      valid_rows <- !is.na(names_extracted) & names_extracted != "" &
        !is.na(surnames_extracted) & surnames_extracted != ""
      
      names_extracted <- names_extracted[valid_rows]
      surnames_extracted <- surnames_extracted[valid_rows]
      
      if (length(names_extracted) > 0 && length(surnames_extracted) > 0) {
        min_rows <- min(length(names_extracted), length(surnames_extracted))
        for (i in 1:min_rows) {
          all_txt_names_pairs[[length(all_txt_names_pairs) + 1]] <-
            c(names_extracted[i], surnames_extracted[i])
        }
      }
    }, error = function(e) {
      warning("Error reading or parsing TXT file '", basename(file_path), "': ", e$message)
    })
  }
  
  # --- 2. Read names from the CSV file into a list of vectors ---
  csv_names_pairs <- list()
  
  if (!dir.exists(csv_folder_path)) {
    warning("CSV folder does not exist: ", csv_folder_path)
    # Even if TXT processing worked, we can't compare without CSV, so return empty.
    return(list())
  }
  
  csv_files <- list.files(csv_folder_path, pattern = "\\.csv$", full.names = TRUE, ignore.case = TRUE)
  if (length(csv_files) == 0) {
    warning("No .csv files found in: ", csv_folder_path)
    return(list())
  }
  if (length(csv_files) > 1) {
    warning("More than one .csv file found in '", csv_folder_path, "'. Reading only the first one: ", basename(csv_files[1]))
  }
  
  csv_file_path <- csv_files[1]
  tryCatch({
    # Read CSV file: no skip, 1st row for headers, comma-separated
    csv_data <- read.csv(csv_file_path, header = TRUE, stringsAsFactors = FALSE, quote = "\"", encoding = "latin1")
    
    # Check if column indices are valid
    num_cols_csv <- ncol(csv_data)
    if (csv_name_col_idx > num_cols_csv || csv_name_col_idx < 1 ||
        csv_surname_col_idx > num_cols_csv || csv_surname_col_idx < 1) {
      warning("Invalid column indices (Name/Surname) for CSV file. File has ", num_cols_csv, " columns. Skipping CSV processing.")
      return(list()) # Return empty if CSV cannot be processed
    }
    
    names_extracted_csv <- csv_data[[csv_name_col_idx]]
    surnames_extracted_csv <- csv_data[[csv_surname_col_idx]]
    
    # Trim whitespace (quotes should already be handled by read.csv(quote="\""))
    names_extracted_csv <- trimws(names_extracted_csv)
    surnames_extracted_csv <- trimws(surnames_extracted_csv)
    
    # Remove any rows where Name or Surname became empty after trimming
    valid_rows_csv <- !is.na(names_extracted_csv) & names_extracted_csv != "" &
      !is.na(surnames_extracted_csv) & surnames_extracted_csv != ""
    
    names_extracted_csv <- names_extracted_csv[valid_rows_csv]
    surnames_extracted_csv <- surnames_extracted_csv[valid_rows_csv]
    
    
    if (length(names_extracted_csv) > 0 && length(surnames_extracted_csv) > 0) {
      min_rows_csv <- min(length(names_extracted_csv), length(surnames_extracted_csv))
      for (i in 1:min_rows_csv) {
        csv_names_pairs[[length(csv_names_pairs) + 1]] <-
          c(names_extracted_csv[i], surnames_extracted_csv[i])
      }
    }
  }, error = function(e) {
    warning("Error reading CSV file '", basename(csv_file_path), "': ", e$message)
    # If CSV reading fails, we can't do a meaningful comparison.
    return(list())
  })
  
  # --- 3. Compare names ---
  # Prepare unique combined strings (lowercase) for efficient comparison
  unique_txt_combined <- unique(lapply(all_txt_names_pairs, function(x) tolower(paste(x, collapse = " "))))
  unique_csv_combined <- unique(lapply(csv_names_pairs, function(x) tolower(paste(x, collapse = " "))))
  
  missing_names_final <- list()
  processed_txt_signatures <- character(0) # To prevent duplicate entries in output list
  
  for (txt_pair in all_txt_names_pairs) {
    txt_combined_lower <- tolower(paste(txt_pair, collapse = " "))
    if (!(txt_combined_lower %in% unique_csv_combined)) {
      if (!(txt_combined_lower %in% processed_txt_signatures)) {
        missing_names_final[[length(missing_names_final) + 1]] <- txt_pair
        processed_txt_signatures <- c(processed_txt_signatures, txt_combined_lower)
      }
    }
  }
  
  # --- 4. Save result to CSV and Print to CMD ---
  
  # Convert the list of name pairs to a data frame for saving
  if (length(missing_names_final) > 0) {
    # Use do.call(rbind, ...) to combine list of vectors into a matrix, then as.data.frame
    # Ensure column names are "Name" and "Surname"
    result_df <- as.data.frame(do.call(rbind, missing_names_final), stringsAsFactors = FALSE)
    colnames(result_df) <- c("Name", "Surname")
  } else {
    # If no missing names, create an empty data frame with the expected columns
    result_df <- data.frame(Name = character(0), Surname = character(0), stringsAsFactors = FALSE)
  }
  
  # Save to CSV
  tryCatch({
    write.csv(result_df, file = output_csv_path, row.names = FALSE)
    message("\nResult saved to: ", output_csv_path)
  }, error = function(e) {
    warning("Could not save results to CSV file '", output_csv_path, "': ", e$message)
  })
  
  # Print to CMD line
  message("\n--- Names present in TXT files but NOT in CSV file ---")
  if (nrow(result_df) > 0) {
    print(result_df)
  } else {
    message("No missing names found.")
  }
  message("-----------------------------------------------------")
  
  return(missing_names_final)
}

jogu_m4_gis <- 'U:/Seafile/Arbeit/Lehre/StudiListenAbgleich/Jogustine/M4GIS'
jogu_m7ed_gis <- 'U:/Seafile/Arbeit/Lehre/StudiListenAbgleich/Jogustine/M7EdGIS'
jogu_karto <- 'U:/Seafile/Arbeit/Lehre/StudiListenAbgleich/Jogustine/Karto'

moodle_m4_gis <- 'U:/Seafile/Arbeit/Lehre/StudiListenAbgleich/Moodle/M4GIS'
moodle_m7ed_gis <- 'U:/Seafile/Arbeit/Lehre/StudiListenAbgleich/Moodle/M7EdGIS'
moodle_karto <- 'U:/Seafile/Arbeit/Lehre/StudiListenAbgleich/Moodle/Karto'

outputCSVpath_m4_gis <- 'U:/Seafile/Arbeit/Lehre/StudiListenAbgleich/outputs/M4GIS_missingNames.csv'
outputCSVpath_m7ed_gis <- 'U:/Seafile/Arbeit/Lehre/StudiListenAbgleich/outputs/M7EdGIS_missingNames.csv'
outputCSVpath_karto <- 'U:/Seafile/Arbeit/Lehre/StudiListenAbgleich/outputs/Karto_missingNames.csv'


# M4 GIS
missingNamesM4 <- compare_names_fixed_cols(jogu_m4_gis,moodle_m4_gis,outputCSVpath_m4_gis)
print(missingNamesM4)

# M7Ed GIS
missingNamesM7Ed <- compare_names_fixed_cols(jogu_m7ed_gis,moodle_m7ed_gis,outputCSVpath_m7ed_gis)
print(missingNamesM7Ed)

# Karto
missingNamesKarto <- compare_names_fixed_cols(jogu_karto,moodle_karto,outputCSVpath_karto)
print(missingNamesKarto)

