import { Component } from '@angular/core';
import { HttpHeaders, HttpClient } from '@angular/common/http'

@Component({
  selector: 'app-forms',
  templateUrl: './forms.component.html',
  styleUrls: ['./forms.component.css']
})
export class FormsComponent {
  question: string = '';
  context: string = '';
  answer: string = '';
  gpt2Question: string = '';
  gpt2Answer: string = '';
  Answer: string = '';
  Question: string = '';
  selectedFile: File | null = null; // Initialize selectedFile with null
  selectedSection: string = 'questionAnswering';

  constructor(private http: HttpClient) {}

  askQuestion() {
    const data = { context: this.context, question: this.question };
    
    const headers = { 'Content-Type': 'application/json' };

    this.http.post('http://127.0.0.1:8000/answer', data, { headers }).subscribe(
      (response: any) => {
        this.answer = response.answer;
      },
      (error) => {
        console.error('Error:', error);
        this.answer = 'An error occurred while fetching the answer.';
      }
    );
  }
    
  askGpt2Question() {
    const data = { question: this.gpt2Question };
    const headers = { 'Content-Type': 'application/json' };
  

    this.http.post('http://localhost:8000/ask-medical-question', data, { headers }).subscribe(
      (response: any) => {
        this.gpt2Answer = response.answer;
      },
      (error) => {
        console.error('Error:', error);
        this.gpt2Answer = 'An error occurred while generating GPT-2 text.';
      }
  );
}
submitForm() {
  if (!this.selectedFile) {
    // Handle the case where no file is selected
    return;
  }

  const formData = new FormData();
  formData.append('file', this.selectedFile, this.selectedFile.name); // Use the File object
  formData.append('question', this.Question); // Include the user's question in the FormData

  // Send a POST request to your API
  this.http.post('http://localhost:8000/document', formData).subscribe(
    (response: any) => {
        this.Answer = response.answer;
      },
      (error) => {
        // Handle errors
        console.error('Error:', error);
      }
    );
  }

  onFileSelected(event: any) {
    this.selectedFile = event.target.files[0];
  }

  changeSelectedSection(section: string) {
    this.selectedSection = section;
  }
}
