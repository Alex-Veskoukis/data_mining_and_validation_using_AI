# AI‑Driven Data Mining Framework for Assessing Decision‑Tree Rule Hiding in Regulated Data


This notebook serves as both a methodological record and analytical report for a large‑scale study into **decision‑tree rule hiding** under modern privacy regulations.  The goal is to characterise the background landscape of decision‑tree applications in industry, identify the types of data features being used, and determine which of those features fall within the scope of privacy laws.  The rationale is that understanding the prevalence and regulatory status of these features informs why methods for hiding or obfuscating decision‑tree rules are becoming increasingly important.

### Methodological overview

1. **Data source selection:** Published scientific articles were chosen as the primary source of real‑world examples because peer‑reviewed publications often describe the use of analytics methods in practice.  A bespoke Python package was developed to mine publicly available article metadata from the Crossref and OpenAlex APIs.  Searches were constructed for twelve industries—  

    *Banking Finance, Healthcare Pharma, Insurance, E‑commerce Retail, Telecom Network Security, Social Media, Education Learning Analytics, IoT Smart Systems, Government Public Administration, Cybersecurity Intrusion Detection, HR Recruitment* and *Transportation Logistics*   
   
    —to capture the breadth of decision‑tree applications.

2. **Article retrieval and merging:** For each industry‑specific query, 1 000 results were retrieved from each source (Crossref and OpenAlex).  The results were merged and deduplicated based on DOIs to create a consolidated corpus of candidate articles.

3. **Relevance filtering with AI:** Because search results can be noisy, a language model was prompted to classify each article as **relevant** or **not relevant** to the application of decision trees using its title, abstract and publication venue.  Only articles flagged as relevant were retained.

4. **Domain validation:** The same AI model was then used to assign each relevant article to one of the twelve target industries (or to *none of the above*), producing a validated domain label for each record.

5. **Feature extraction and validation:** A key objective is to understand which data features are used in decision‑tree models.  The AI was instructed to extract from each abstract any feature names mentioned in connection with training a decision tree.  A second validation step confirmed that each extracted feature was indeed used for model training and not merely mentioned in passing.

6. **Attribute‑class assignment:** Each validated feature was mapped by the AI to one of thirteen privacy‑aware ***attribute classes***:   
    *Identifier_PII*, *Contact_Info*, *Device_OnlineID*, *Biometric*, *Location_IoT*, *Health_Clinical*, *Financial*, *Child_Data*, *Demographic*, *Behavioural*, *Environmental*, *Operational_Business* or *Other*.  
   
    These classes align with common privacy taxonomies and are used in later regulatory matching.

7. **Regulatory text mining:** Official texts of major privacy laws (e.g., GDPR, HIPAA, CCPA) were downloaded.  The AI was instructed to identify passages and paragraphs that mention, regulate or otherwise pertain to any of the thirteen attribute classes listed above.  This produced a mapping from attribute classes to regulatory excerpts.

8. **Regulation–feature linking:** By cross‑referencing the attribute class of each feature with the regulatory passages collected, we created a feature–regulation table.  For each feature, the AI assessed whether it is **regulated** or **not regulated** under the relevant law and assigned a confidence level (high, medium or low).  This step recognises that legal interpretation can be ambiguous; the confidence score captures the strength of the AI’s assessment.

9. **Final dataset and analysis scope:** Merge and harmonise the validated features with the regulation‑mapping tables, deduplicate records and align each feature with its attribute class and regulation status. From this point onward, unless explicitly stated otherwise, **the analytic corpus is limited to those features that the AI assessed as *Regulated* with *High* confidence**. In other words, we analyse only high‑confidence regulated features; items with Medium or Low confidence are omitted from the main analyses and appear only in sensitivity checks.



## Regulatory Information for Context

The dataset does not contain the enactment years of the regulations. The following table lists the official regulations and the enactment years, sourced from authoritative references (citations indicated for validation):

| Regulation | Enactment year | Source |
|-----------|---------------|-------|
| HIPAA | 1996 | Signed into law on 21 August 1996【617437092921950†L114-L116】 |
| HITECH | 2009 | Signed into law on 17 February 2009【316138370182581†L209-L216】 |
| CCPA | 2018 | Passed in 2018【89971892187549†L199-L201】 |
| CPRA | 2020 | Approved by voters on 3 November 2020【943854279962298†L89-L96】 |
| GDPR | 2018 | Became enforceable on 25 May 2018【973256186925503†L21-L31】 |
| ePrivacy Directive | 2002 | Adopted in 2002 and amended in 2009【765434439357368†L114-L119】 |
| NIS2 | 2023 | Entered into force in January 2023【6750382137667†L125-L127】 |
| PSD2 | 2016 | Applied from 12 January 2016【724292858980272†L195-L197】 |
| EU eHealth Network | 2011 | Established under Directive 2011/24/EU【222916198544407†L72-L75】 |
| GLBA | 1999 | Signed into law in November 1999【329363762707909†L43-L46】 |
| COPPA | 1998 | Signed into law on 21 October 1998【227641286164990†L415-L421】 |
| FERPA | 1974 | Enacted in 1974. |
| ECPA | 1986 | Enacted in 1986. |
