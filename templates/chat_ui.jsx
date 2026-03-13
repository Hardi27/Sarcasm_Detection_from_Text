const { useEffect, useRef, useState } = React;

    function ChatApp() {
      const [inputText, setInputText] = useState('');
      const [messages, setMessages] = useState([
        {
          sender: 'bot',
          text: 'Type a sentence and I will predict whether it is sarcastic or not sarcastic.'
        }
      ]);
      const [recording, setRecording] = useState(false);
      const chatRef = useRef(null);
      const recognitionRef = useRef(null);

      useEffect(() => {
        const token = localStorage.getItem('access_token');
        if (!token) {
          window.location.href = '/login';
          return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
          const rec = new SpeechRecognition();
          rec.lang = 'en-US';
          rec.continuous = false;
          rec.interimResults = false;

          rec.onstart = () => setRecording(true);
          rec.onend = () => setRecording(false);
          rec.onerror = () => setRecording(false);
          rec.onresult = (event) => {
            const transcript = event.results?.[0]?.[0]?.transcript || '';
            setInputText(transcript);
          };

          recognitionRef.current = rec;
        }
      }, []);

      useEffect(() => {
        if (chatRef.current) {
          chatRef.current.scrollTop = chatRef.current.scrollHeight;
        }
      }, [messages]);

      function logout() {
        localStorage.removeItem('access_token');
        localStorage.removeItem('username');
        window.location.href = '/login';
      }

      function startVoice() {
        if (!recognitionRef.current) {
          alert('Speech recognition not supported in this browser.');
          return;
        }
        recognitionRef.current.start();
      }

      function countMeaningfulWords(text) {
        return text
          .replace(/http\S+|www\S+/g, ' ')
          .replace(/@\w+/g, ' ')
          .replace(/#/g, ' ')
          .replace(/[^\w\s]/g, ' ')
          .trim()
          .split(/\s+/)
          .filter(Boolean).length;
      }

      function splitSentencesForBatch(text) {
        return text
          .split(/\r?\n+/)
          .map((line) => line.trim())
          .filter(Boolean);
      }

      async function checkSarcasm() {
        const message = inputText.trim();
        if (!message) return;

        const batchLines = splitSentencesForBatch(message);
        const isBatchInput = batchLines.length > 1;

        const minWords = 3;
        if (!isBatchInput && countMeaningfulWords(message) < minWords) {
          setMessages((prev) => [
            ...prev,
            { sender: 'user', text: message },
            { sender: 'bot', text: 'Not enough context. Please enter at least 3 words for a reliable prediction.' }
          ]);
          setInputText('');
          return;
        }

        const token = localStorage.getItem('access_token');
        setMessages((prev) => [...prev, { sender: 'user', text: message }, { sender: 'bot', text: 'Analyzing...', pending: true }]);
        setInputText('');

        try {
          const headers = { 'Content-Type': 'application/json' };
          if (token) headers.Authorization = 'Bearer ' + token;

          const endpoint = isBatchInput ? '/api/predict/batch' : '/api/predict';
          const payload = isBatchInput ? { texts: batchLines } : { text: message };

          const response = await fetch(endpoint, {
            method: 'POST',
            headers,
            body: JSON.stringify(payload)
          });

          const data = await response.json();
          let botText;

          if (!response.ok) {
            const apiError = data.error || data.msg || data.message || data.detail || 'Failed to predict';
            botText = `Error: ${apiError}`;
          } else if (isBatchInput) {
            const rows = Array.isArray(data.items) ? data.items : [];
            const lines = rows.map((item, idx) => {
              if (item.error) {
                return `${idx + 1}. [Invalid] ${item.text || ''}\n   reason: ${item.error}`;
              }
              const tag = item.label || 'UNKNOWN';
              const sarcasticPct = typeof item.sarcastic_percentage === 'number'
                ? `${item.sarcastic_percentage.toFixed(2)}%`
                : 'N/A';
              const nonSarcasticPct = typeof item.non_sarcastic_percentage === 'number'
                ? `${item.non_sarcastic_percentage.toFixed(2)}%`
                : 'N/A';
              const conclusion = item.conclusion || `The sentence is ${item.is_sarcasm ? '' : 'not '}sarcastic.`;
              return `${idx + 1}. [${tag}] ${item.text}\n   sarcastic: ${sarcasticPct} | non sarcastic: ${nonSarcasticPct}\n   ${conclusion}`;
            });
            botText =
              `Batch processed: ${data.count || rows.length}\n\n` +
              lines.join('\n');
          } else if (data.needs_more_context) {
            botText = data.message || 'Not enough context. Please enter a longer sentence.';
          } else {
            const label = data.label || 'UNKNOWN';
            const sarcasticPct = typeof data.sarcastic_percentage === 'number'
              ? `${data.sarcastic_percentage.toFixed(2)}%`
              : 'N/A';
            const nonSarcasticPct = typeof data.non_sarcastic_percentage === 'number'
              ? `${data.non_sarcastic_percentage.toFixed(2)}%`
              : 'N/A';
            const conclusion = data.conclusion || `The sentence is ${data.is_sarcasm ? '' : 'not '}sarcastic.`;
            botText = `[${label}]\nsarcastic: ${sarcasticPct}\nnon sarcastic: ${nonSarcasticPct}\n${conclusion}`;
          }

          setMessages((prev) => {
            const list = [...prev];
            const loadingIndex = list.findIndex((item) => item.pending);
            if (loadingIndex >= 0) {
              list[loadingIndex] = { sender: 'bot', text: botText };
            } else {
              list.push({ sender: 'bot', text: botText });
            }
            return list;
          });
        } catch (e) {
          setMessages((prev) => {
            const list = [...prev];
            const loadingIndex = list.findIndex((item) => item.pending);
            const fallback = { sender: 'bot', text: 'Server error. Please try again.' };
            if (loadingIndex >= 0) {
              list[loadingIndex] = fallback;
            } else {
              list.push(fallback);
            }
            return list;
          });
        }
      }

      function onKeyDown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
          event.preventDefault();
          checkSarcasm();
        }
      }

      return (
        <>
          <button className="logout-btn" onClick={logout}>Logout</button>

            <div className="chat-box">
            <div className="chat-header">Sarcasm Detection</div>
            <div className="chat-messages" ref={chatRef}>
              {messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.sender}`}>
                  {msg.text}
                </div>
              ))}
            </div>

            <div className="chat-input">
              <div className="input-wrapper">
                <textarea
                  placeholder="Type one sentence, or paste many lines (one per line)"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={onKeyDown}
                ></textarea>
                <button
                  className={`mic-btn ${recording ? 'recording' : ''}`}
                  onClick={startVoice}
                  type="button"
                  title="Voice input"
                >
                  <svg className="mic-icon" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v5a3 3 0 0 0 3 3z"></path>
                    <path d="M17 11a1 1 0 1 0-2 0 3 3 0 1 1-6 0 1 1 0 1 0-2 0 5 5 0 0 0 4 4.9V19H9a1 1 0 1 0 0 2h6a1 1 0 1 0 0-2h-2v-3.1A5 5 0 0 0 17 11z"></path>
                  </svg>
                </button>
              </div>
              <button className="send-btn" onClick={checkSarcasm} type="button">Send</button>
            </div>
          </div>
        </>
      );
    }

    ReactDOM.createRoot(document.getElementById('root')).render(<ChatApp />);

