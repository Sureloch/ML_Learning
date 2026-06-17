import { useState } from 'react'
import { useNavigate } from 'react-router-dom'


function LoginPage ({ setToken , setID}){
    const [email, setEmail] =  useState('');
    const [password, setPassword] =  useState('');
    const [loginStatus, setLoginStatus] =  useState(false);
    const navigate = useNavigate()
    const inputInfo = async () => {
      const response = await fetch(`http://localhost:8000/auth/login`,
        {
          method : 'POST',
          headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            email : email,
            password : password
          })
        });
    const data = await response.json()
    if (response.ok)
      setLoginStatus(true)
      setToken(data.access_token)
      setID(data.id)
      navigate('/home')
}
    return(
      <div>
          <input 
          value={email} 
          onChange={(e) => setEmail(e.target.value)} 
          placeholder="Email"
          />
          <input
          type='password' 
          value={password} 
          onChange={(e) => setPassword(e.target.value)} 
          placeholder="Password"
          />
          <button 
          onClick={inputInfo}
          placeholder = "Login">
          Submit
          </button>
      </div>

    )
}
export default LoginPage