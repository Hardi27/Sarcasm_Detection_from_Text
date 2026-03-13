const { useState } = React;

    function LoginApp() {
      const [email, setEmail] = useState("");
      const [password, setPassword] = useState("");
      const [error, setError] = useState("");
      const [loading, setLoading] = useState(false);

      async function handleSubmit(event) {
        event.preventDefault();
        setError("");

        const trimmedEmail = email.trim();
        if (!trimmedEmail || !password) {
          setError("Please fill both fields.");
          return;
        }

        setLoading(true);
        try {
          const response = await fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: trimmedEmail, password })
          });

          const data = await response.json();
          if (!response.ok) {
            setError(data.error || 'Login failed');
            return;
          }

          localStorage.setItem('access_token', data.access_token);
          localStorage.setItem('username', data.username);
          window.location.href = '/';
        } catch (e) {
          setError('Server error. Please try again.');
        } finally {
          setLoading(false);
        }
      }

      return (
        <div className="card">
          <h2>Login</h2>
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
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            <button type="submit" disabled={loading}>
              {loading ? 'Logging in...' : 'Login'}
            </button>
          </form>

          <div className="small">
            Do not have an account?{' '}
            <span className="link" onClick={() => window.location.href = '/signup'}>Sign up</span>
          </div>
        </div>
      );
    }

    ReactDOM.createRoot(document.getElementById('root')).render(<LoginApp />);

