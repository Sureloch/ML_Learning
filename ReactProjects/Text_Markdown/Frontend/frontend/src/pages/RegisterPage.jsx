import { useState } from 'react'
import { useNavigate } from 'react-router-dom'


function RegisterPage (){
    const [email, setEmail] =  useState('');
    const [password, setPassword] =  useState('');
    const [name, setName] =  useState('');
    const [loginStatus, setLoginStatus] =  useState(false);
    const navigate = useNavigate()
    const inputInfo = async () => {
      const response = await fetch(`http://localhost:8000/auth/register`,
        {
          method : 'POST',
          headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            email : email,
            name : name,
            password : password
          })
        });
    const data = await response.json()
    if (response.ok)
      setLoginStatus(true)
      navigate('/login')
    }
    return(
      <div>
          <input 
          value={email} 
          onChange={(e) => setEmail(e.target.value)} 
          placeholder="Email"
          />
          <input 
          value={name} 
          onChange={(e) => setName(e.target.value)} 
          placeholder="Name"
          />
          <input
          type='password' 
          value={password} 
          onChange={(e) => setPassword(e.target.value)} 
          placeholder="Password"
          />
          <button onClick={inputInfo}>Submit</button>
      </div>

    )
}
export default RegisterPage