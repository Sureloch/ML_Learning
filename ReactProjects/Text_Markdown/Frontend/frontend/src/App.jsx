import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { useState } from 'react'
import RegisterPage from './pages/RegisterPage'
import LoginPage from './pages/LoginPage'
import HomePage from './pages/HomePage'

function App() {
    const [token, setToken] = useState(null)
    const [id, setID] = useState(null)
    return (
        <BrowserRouter>
            <Routes>
                <Route path="/register" element={<RegisterPage  />} />
                <Route path="/login" element={<LoginPage setToken = {setToken} setID = {setID} />} />
                <Route path="/home" element = {<HomePage token = {token} id = {id}/>}/>
            </Routes>
        </BrowserRouter>
    )
}
export default App