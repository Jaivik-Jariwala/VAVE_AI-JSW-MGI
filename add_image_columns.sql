-- SQL Migration Script: Add current_scenario_image and proposal_scenario_image columns
-- Run this script on your PostgreSQL database to add the new image columns

-- Add current_scenario_image column
ALTER TABLE ideas 
ADD COLUMN IF NOT EXISTS current_scenario_image TEXT;

-- Add proposal_scenario_image column  
ALTER TABLE ideas 
ADD COLUMN IF NOT EXISTS proposal_scenario_image TEXT;

-- Add comments for documentation
COMMENT ON COLUMN ideas.current_scenario_image IS 'Path to the current scenario image for the idea';
COMMENT ON COLUMN ideas.proposal_scenario_image IS 'Path to the proposal scenario image for the idea';

-- Verify columns were added
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'ideas' 
AND column_name IN ('current_scenario_image', 'proposal_scenario_image');

