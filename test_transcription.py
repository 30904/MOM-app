from services.transcription_service import TranscriptionService
import os

def test_transcription():
    try:
        ts = TranscriptionService()
        file_path = os.path.join('uploads', 'IMF_Clears_2.3_Billion_for_Pakistan_Missiles_Follow_Hours_Later_Vantage_with_Palki_Sharma_N18G.mp3')
        print(f"Testing transcription of file: {file_path}")
        text = ts.transcribe_file(file_path)
        print('Transcription result:', text)
    except Exception as e:
        print('Error:', str(e))

if __name__ == "__main__":
    test_transcription() 