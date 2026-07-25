import {
  Component,
  NgZone,
  ChangeDetectorRef
} from '@angular/core';

import {
  HttpClient
} from '@angular/common/http';

import {
  FormsModule
} from '@angular/forms';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrls: ['./app.css']
})
export class App {

  userMessage: string = '';
  aiResponse: string = '';
  isListening: boolean = false;

  constructor(
    private http: HttpClient,
    private zone: NgZone,
    private cdr: ChangeDetectorRef
  ) {}

  startListening() {

    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {

      alert(
        'Speech Recognition not supported in this browser.'
      );

      return;
    }

    const recognition =
      new SpeechRecognition();

    recognition.lang = 'en-US';
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {

      this.zone.run(() => {

        this.isListening = true;
        this.cdr.detectChanges();

      });

      console.log('Listening...');
    };

    recognition.onresult = (event: any) => {

      const text =
        event.results[0][0].transcript;

      console.log('Recognized:', text);

      this.zone.run(() => {

        this.userMessage = text;

        console.log(
          'userMessage =',
          this.userMessage
        );

        this.cdr.detectChanges();

        this.sendMessage();

      });
    };

    recognition.onerror = (event: any) => {

      console.error(
        'Speech Error:',
        event.error
      );

      this.zone.run(() => {

        this.isListening = false;
        this.cdr.detectChanges();

      });
    };

    recognition.onend = () => {

      console.log(
        'Recognition ended'
      );

      this.zone.run(() => {

        this.isListening = false;
        this.cdr.detectChanges();

      });
    };

    recognition.start();
  }

  sendMessage() {

    if (!this.userMessage.trim()) {
      return;
    }

    console.log(
      'Sending:',
      this.userMessage
    );

    this.http.post<any>(
      'http://127.0.0.1:8000/chat',
      {
        message: this.userMessage
      }
    )
    .subscribe({

      next: (res) => {

        console.log(
          'Response:',
          res
        );

        this.zone.run(() => {

          this.aiResponse =
            res.response;

          this.cdr.detectChanges();

        });
      },

      error: (err) => {

        console.error(
          'Backend Error:',
          err
        );

        this.zone.run(() => {

          this.aiResponse =
            'Could not connect to backend.';

          this.cdr.detectChanges();

        });
      }
    });
  }

  test() {

    this.userMessage =
      'TEST MESSAGE';

    this.aiResponse =
      'TEST RESPONSE';

    this.cdr.detectChanges();
  }
}