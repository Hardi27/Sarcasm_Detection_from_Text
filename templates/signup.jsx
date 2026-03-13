const { useState } = React;

    function SignupApp() {
      const [email, setEmail] = useState("");
      const [username, setUsername] = useState("");
      const [password, setPassword] = useState("");
      const [error, setError] = useState("");
      const [loading, setLoading] = useState(false);

      async function handleSubmit(event) {
        event.preventDefault();
        setError("");

        const trimmedEmail = email.trim();
        const trimmedUsername = username.trim();

        if (!trimmedEmail || !trimmedUsername || !password) {
          setError('All fields are required.');
          return;
        }
        if (password.length < 6) {
          setError('Password must be at least 6 characters.');
          return;
        }

        setLoading(true);
        try {
          const response = await fetch('/api/signup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: trimmedUsername, email: trimmedEmail, password })
          });

          const data = await response.json();
          if (!response.ok) {
            setError(data.error || 'Signup failed');
            return;
          }

          alert('Account created successfully. Please login.');
          window.location.href = '/login';
        } catch (e) {
          setError('Server error. Please try again.');
        } finally {
          setLoading(false);
        }
      }

      return (
        <div className="card">
          <h2>Create Account</h2>
          <div className="error">{error}</div>

          <form onSubmit={handleSubmit}>
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoFocus
            />
            <input
              type="text"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
            <input
              type="password"
              placeholder="Password (min 6 chars)"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            <button type="submit" disabled={loading}>
              {loading ? 'Creating...' : 'Sign Up'}
            </button>
          </form>

          <div className="small">
            Already have an account?{' '}
            <span className="link" onClick={() => window.location.href = '/login'}>Login</span>
          </div>
        </div>
      );
    }

    ReactDOM.createRoot(document.getElementById('root')).render(<SignupApp />);

